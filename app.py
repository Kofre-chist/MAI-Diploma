from flask import Flask, render_template, request, redirect, url_for, jsonify
from models import db, Project, Task, Resource, TaskResourceUsage, TaskPrecedence
import pyomo.environ as pyo
from pyomo.environ import SolverFactory

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:213790@localhost:5432/rcpsp_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

@app.route('/')
def index():
    return render_template('index.html')

def build_resource_schedule(schedule, resources):
    """Преобразует расписание задач → расписание использования ресурсов."""
    res_name = {r.id: r.name for r in resources}

    resource_schedule = []
    for entry in schedule:
        task_id = entry["task"]
        for u in (TaskResourceUsage.query
                  .filter_by(task_id=task_id)
                  .filter(TaskResourceUsage.amount > 0)):
            resource_schedule.append({
                "resource_id": u.resource_id,
                "resource_name": res_name[u.resource_id],
                "task_name":    entry["name"],
                "amount":       u.amount,
                "start":        entry["start"],
                "end":          entry["end"]
            })
    return resource_schedule

@app.route('/new_project', methods=['GET', 'POST'])
def new_project():
    if request.method == 'POST':
        project_selection = request.form.get('project_selection')
        project_name_new = request.form.get('project_name_new', "").strip()
        if project_name_new:
            project = Project(name=project_name_new)
            db.session.add(project)
            db.session.commit()
        else:
            try:
                project_id = int(project_selection)
                project = Project.query.get(project_id)
                if not project:
                    return "Выбран неверный проект", 400
            except:
                return "Неверный выбор проекта", 400

        resource_names = request.form.getlist('resource_name[]')
        resource_caps = request.form.getlist('resource_capacity[]')
        resources = []
        for rn, rc in zip(resource_names, resource_caps):
            if rn.strip():
                r = Resource(name=rn.strip(), capacity=int(rc), project_id=project.id)
                db.session.add(r)
                resources.append(r)
        db.session.commit()
        
        task_names = request.form.getlist('task_name[]')
        task_durations = request.form.getlist('task_duration[]')
        tasks = []
        for tn, td in zip(task_names, task_durations):
            if tn.strip():
                t = Task(project_id=project.id, name=tn.strip(), duration=int(td))
                db.session.add(t)
                tasks.append(t)
        db.session.commit()
        
        usage_tasks = request.form.getlist('usage_task[]')
        usage_resources = request.form.getlist('usage_resource[]')
        usage_amounts = request.form.getlist('usage_amount[]')
        given_pairs = set()
        for ut, ur, ua in zip(usage_tasks, usage_resources, usage_amounts):
            try:
                task_index = int(ut)
                res_id = int(ur)
                given_pairs.add( (task_index, res_id) )
                if 0 <= task_index - 1 < len(tasks):
                    t = tasks[task_index - 1]
                    usage = TaskResourceUsage(task_id=t.id, resource_id=res_id, amount=int(ua))
                    db.session.add(usage)
            except:
                continue
        db.session.commit()
        for idx, t in enumerate(tasks, start=1):
            for r in resources:
                if (idx, r.id) not in given_pairs:
                    usage = TaskResourceUsage(task_id=t.id, resource_id=r.id, amount=0)
                    db.session.add(usage)
        db.session.commit()
        
        preds = request.form.getlist('predecessor_id[]')
        succs = request.form.getlist('successor_id[]')
        for p_val, s_val in zip(preds, succs):
            if p_val.strip() and s_val.strip():
                try:
                    pred_idx = int(p_val)
                    succ_idx = int(s_val)
                    if 1 <= pred_idx <= len(tasks) and 1 <= succ_idx <= len(tasks):
                        p_entry = TaskPrecedence(predecessor_id=tasks[pred_idx - 1].id,
                                                 successor_id=tasks[succ_idx - 1].id)
                        db.session.add(p_entry)
                except:
                    continue
        db.session.commit()
        
        solution_data = optimize_project(project.id)

        resource_schedule = build_resource_schedule(solution_data["schedule"], resources)
        
        return render_template('all_in_one.html',
                               mode='results',
                               project=project,
                               resources=resources,
                               tasks=tasks,
                               schedule=solution_data.get('schedule'),
                               makespan=solution_data.get('makespan'),
                               status=solution_data.get('status'),
                               resource_schedule=resource_schedule)
    else:
        projects = Project.query.all()
        resources = Resource.query.all()
        return render_template('all_in_one.html', mode='input', projects=projects, resources=resources)

@app.route('/projects')
def projects():
    all_projects = Project.query.all()
    return render_template('projects.html', projects=all_projects)

@app.route('/delete_project/<int:project_id>', methods=['POST'])
def delete_project(project_id):
    project = Project.query.get(project_id)
    if not project:
        return "Проект не найден", 404

    for task in project.tasks:
        TaskResourceUsage.query.filter_by(task_id=task.id).delete()
        TaskPrecedence.query.filter(
            (TaskPrecedence.predecessor_id == task.id) | (TaskPrecedence.successor_id == task.id)
        ).delete()
        db.session.delete(task)

    db.session.delete(project)
    db.session.commit()
    return redirect(url_for('projects'))

@app.route('/view_project/<int:project_id>')
def view_project(project_id):
    project = Project.query.get(project_id)
    if not project:
        return "Проект не найден", 404
    solution_data = optimize_project(project.id)
    resources = Resource.query.filter_by(project_id=project.id).all()
    tasks = project.tasks
    resource_schedule = build_resource_schedule(solution_data["schedule"], resources)
    return render_template('all_in_one.html',
                           mode='results',
                           project=project,
                           resources=resources,
                           tasks=tasks,
                           schedule=solution_data.get('schedule'),
                           makespan=solution_data.get('makespan'),
                           status=solution_data.get('status'),
                           resource_schedule=resource_schedule)

@app.route('/compare_projects', methods=['GET', 'POST'])
def compare_projects():
    if request.method == 'POST':
        selected_ids = request.form.getlist('project_ids')
        if not selected_ids:
            return "Выберите хотя бы один проект", 400

        all_schedule = []
        tmp_makespan = 0
        for pid in selected_ids:
            pid = int(pid)
            solution = optimize_project(pid)
            tmp_makespan = max(tmp_makespan, solution.get('makespan'))
            project = Project.query.get(pid)
            if solution["status"] != "optimal":
                continue

            for entry in solution["schedule"]:
                entry["project_name"] = project.name
                all_schedule.append(entry)
        
            project_names = list({s["project_name"] for s in all_schedule})
            palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            color_map = {name: palette[i % len(palette)] 
                for i, name in enumerate(project_names)}

        return render_template("compare_projects.html", schedule=all_schedule, makespan=tmp_makespan, color_map=color_map)

    all_projects = Project.query.all()
    return render_template("select_projects.html", projects=all_projects)

@app.route('/edit_project/<int:project_id>', methods=['GET', 'POST'])
def edit_project(project_id):
    """Редактирование проекта + вывод результатов оптимизации."""
    project = Project.query.get_or_404(project_id)

    if request.method == 'POST':
        project.name = request.form.get('project_name', '').strip()

        existing_res = {r.id: r for r in Resource.query.filter_by(project_id=project.id)}
        f_res_ids   = request.form.getlist('resource_id[]')
        f_res_names = request.form.getlist('resource_name[]')
        f_res_caps  = request.form.getlist('resource_capacity[]')

        for rid, nm, cap in zip(f_res_ids, f_res_names, f_res_caps):
            nm = nm.strip()
            if not nm:
                continue
            if rid:
                res = existing_res.pop(int(rid), None)
                if res:
                    res.name = nm
                    res.capacity = int(cap)
            else:
                db.session.add(Resource(
                    name=nm, capacity=int(cap), project_id=project.id))

        for r in existing_res.values():
            TaskResourceUsage.query.filter_by(resource_id=r.id).delete(synchronize_session=False)
            db.session.delete(r)
        db.session.flush()

        existing_tasks = {t.id: t for t in project.tasks}
        f_task_ids   = request.form.getlist('task_id[]')
        f_task_names = request.form.getlist('task_name[]')
        f_task_durs  = request.form.getlist('task_duration[]')

        idx_map: dict[int, Task] = {}
        for idx, (tid, nm, du) in enumerate(zip(f_task_ids, f_task_names, f_task_durs), start=1):
            nm = nm.strip()
            if not nm:
                continue
            if tid:
                task = existing_tasks.pop(int(tid), None)
                if task:
                    task.name = nm
                    task.duration = int(du)
            else:
                task = Task(project_id=project.id, name=nm, duration=int(du))
                db.session.add(task)
            idx_map[idx] = task

        for t in existing_tasks.values():
            TaskResourceUsage.query.filter_by(task_id=t.id).delete(synchronize_session=False)
            TaskPrecedence.query.filter(
                (TaskPrecedence.predecessor_id == t.id) |
                (TaskPrecedence.successor_id == t.id)
            ).delete(synchronize_session=False)
            db.session.delete(t)
        db.session.flush()

        TaskResourceUsage.query.filter(
            TaskResourceUsage.task_id.in_([t.id for t in idx_map.values()])
        ).delete(synchronize_session=False)

        for row_idx, res_id, amt in zip(request.form.getlist('usage_task[]'),
                                        request.form.getlist('usage_resource[]'),
                                        request.form.getlist('usage_amount[]')):
            try:
                task_obj = idx_map.get(int(row_idx))
                if task_obj:
                    db.session.add(TaskResourceUsage(
                        task_id=task_obj.id,
                        resource_id=int(res_id),
                        amount=int(amt)))
            except ValueError:
                continue

        TaskPrecedence.query.filter(
            TaskPrecedence.predecessor_id.in_([t.id for t in idx_map.values()])
        ).delete(synchronize_session=False)

        for p_idx, s_idx in zip(request.form.getlist('predecessor_id[]'),
                                request.form.getlist('successor_id[]')):
            try:
                p_task = idx_map.get(int(p_idx))
                s_task = idx_map.get(int(s_idx))
                if p_task and s_task:
                    db.session.add(TaskPrecedence(
                        predecessor_id=p_task.id, successor_id=s_task.id))
            except ValueError:
                continue

        db.session.commit()

        solution_data = optimize_project(project.id)

        usage_data: dict[tuple[int, int], int] = {}
        for u in TaskResourceUsage.query.filter(
                TaskResourceUsage.task_id.in_([t.id for t in idx_map.values()])):
            key = (u.task_id, u.resource_id)
            if key not in usage_data or u.amount > usage_data[key]:
                usage_data[key] = u.amount
        
        resources = Resource.query.filter_by(project_id=project.id).all()
        resource_schedule = build_resource_schedule(solution_data["schedule"], resources)

        return render_template(
            'edit_project.html',
            mode='results',
            project=project,
            resources=Resource.query.filter_by(project_id=project.id).all(),
            tasks=list(idx_map.values()),
            usage_data=usage_data,
            solution=solution_data,
            resource_schedule=resource_schedule)

    resources = Resource.query.filter_by(project_id=project.id).all()
    tasks     = project.tasks
    task_index_map = {t.id: idx for idx, t in enumerate(tasks, 1)}

    usage_data: dict[tuple[int, int], int] = {}
    for u in TaskResourceUsage.query.join(Task).filter(Task.project_id == project.id):
        key = (u.task_id, u.resource_id)
        if key not in usage_data or u.amount > usage_data[key]:
            usage_data[key] = u.amount

    precedences = TaskPrecedence.query.filter(
        TaskPrecedence.predecessor_id.in_([t.id for t in tasks]),
        TaskPrecedence.successor_id.in_([t.id for t in tasks])
    ).all()

    resources = Resource.query.filter_by(project_id=project.id).all()

    return render_template(
        'edit_project.html',
        mode='input',
        project=project,
        resources=resources,
        tasks=tasks,
        usage_data=usage_data,
        precedences=precedences,
        task_index_map=task_index_map,
        resource_schedule=[])

def optimize_project(project_id):
    project = Project.query.get(project_id)
    if not project:
        return {"status": "error", "msg": "Project not found"}
    tasks = project.tasks
    resources = Resource.query.filter_by(project_id=project_id).all()
    precedences = (TaskPrecedence.query
                   .join(Task, TaskPrecedence.successor_id == Task.id)
                   .filter(Task.project_id == project_id).all())
    
    task_ids = [t.id for t in tasks]
    durations = {t.id: t.duration for t in tasks}
    resource_caps = {r.id: r.capacity for r in resources}
    
    usage_map = {}
    for t in tasks:
        for u in t.resource_usages:
            usage_map[(t.id, u.resource_id)] = u.amount
    
    precedence_list = [(p.predecessor_id, p.successor_id) for p in precedences]
    task_names = {t.id: t.name for t in tasks}
    
    return run_optimization_multi(task_ids, durations, resource_caps, usage_map, precedence_list, task_names)

def run_optimization_multi(task_ids, durations, resource_caps, usage_map, precedence_list, task_names):
    model = pyo.ConcreteModel("MultiResourceRCPSP")
    model.TASKS = pyo.Set(initialize=task_ids)
    model.RES = pyo.Set(initialize=list(resource_caps.keys()))
    horizon = sum(durations.values())
    model.TIME = pyo.RangeSet(0, horizon)
    model.PRECEDENCE = pyo.Set(dimen=2, initialize=precedence_list)
    
    model.x = pyo.Var(model.TASKS, model.TIME, domain=pyo.Binary)
    model.M = pyo.Var(domain=pyo.NonNegativeReals)
    
    def start_once_rule(m, t):
        return sum(m.x[t, tau] for tau in m.TIME) == 1
    model.start_once = pyo.Constraint(model.TASKS, rule=start_once_rule)
    
    def capacity_rule(m, r, tau):
        components = [
            usage_map.get((t, r), 0) * m.x[t, start_t]
            for t in m.TASKS
            for start_t in m.TIME
            if start_t <= tau < start_t + durations[t]
        ]
        if not components:
            return pyo.Constraint.Skip
        return sum(components) <= resource_caps[r]
    model.capacity_cons = pyo.Constraint(model.RES, model.TIME, rule=capacity_rule)
    
    def precedence_rule(m, t1, t2):
        return sum(tau * m.x[t2, tau] for tau in m.TIME) >= \
               sum((kk + durations[t1]) * m.x[t1, kk] for kk in m.TIME)
    model.precedence_cons = pyo.Constraint(model.PRECEDENCE, rule=precedence_rule)
    
    def makespan_rule(m, t):
        return m.M >= sum((tau + durations[t]) * m.x[t, tau] for tau in m.TIME)
    model.makespan_cons = pyo.Constraint(model.TASKS, rule=makespan_rule)
    
    model.obj = pyo.Objective(expr=model.M, sense=pyo.minimize)
    
    solver_path = r"C:\GLPK\w64\glpsol.exe"
    solver = SolverFactory("glpk", executable=solver_path)
    result = solver.solve(model, tee=False)
    
    if (result.solver.status == pyo.SolverStatus.ok) and \
       (result.solver.termination_condition == pyo.TerminationCondition.optimal):
        schedule = []
        for t in task_ids:
            for tau in model.TIME:
                if pyo.value(model.x[t, tau]) > 0.5:
                    schedule.append({
                        "task": t,
                        "start": tau,
                        "end": tau + durations[t],
                        "name": task_names[t]
                    })
                    break
        return {"status": "optimal", "makespan": pyo.value(model.M), "schedule": schedule}
    else:
        return {"status": "infeasible_or_no_solution"}

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)