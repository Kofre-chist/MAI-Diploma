from flask import (Flask, render_template, request, redirect,
                   url_for, jsonify)

from models import (db, Portfolio, Project, Task, Resource,
                    TaskResourceUsage, TaskPrecedence)

import pyomo.environ as pyo
from pyomo.environ import SolverFactory
from sqlalchemy import select, exists, and_, or_

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:213790@localhost:5432/rcpsp_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def compatible_project_query(pf_id: int):
    
    alien_res_subq = (
        select(1)
        .select_from(Task)
        .join(TaskResourceUsage, Task.id == TaskResourceUsage.task_id)
        .join(Resource, Resource.id == TaskResourceUsage.resource_id)
        .where(
            and_(
                Task.project_id == Project.id,
                Resource.portfolio_id.isnot(None),
                Resource.portfolio_id != pf_id
            )
        )
        .limit(1)
    )

    return Project.query.filter(
        or_(Project.portfolio_id.is_(None), Project.portfolio_id != pf_id)
    ).filter(~exists(alien_res_subq))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/portfolios')
def portfolios():
    return render_template('portfolios.html', portfolios=Portfolio.query.all())


@app.route('/new_portfolio', methods=['GET', 'POST'])
def new_portfolio():
    if request.method == 'POST':
        name = request.form.get('portfolio_name', '').strip()
        if not name:
            return "Имя не указано", 400
        pf = Portfolio(name=name)
        db.session.add(pf)
        db.session.commit()
        return redirect(url_for('portfolios'))
    return render_template('new_portfolio.html')


@app.route('/view_portfolio/<int:pf_id>')
def view_portfolio(pf_id):
    pf = Portfolio.query.get_or_404(pf_id)

    solution = optimize_portfolio(pf_id)

    proj_names = {s["name"].split(":", 1)[0] for s in solution["schedule"]}
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    color_map = {name: palette[i % len(palette)]
                 for i, name in enumerate(sorted(proj_names))}
    
    res_port = Resource.query.filter_by(portfolio_id=pf_id)
    res_proj = Resource.query.filter(Resource.project_id.in_([p.id for p in pf.projects]))
    resources = list(res_port) + list(res_proj)

    resource_schedule = build_resource_schedule(solution['schedule'], resources)

    unassigned_projects = compatible_project_query(pf_id) \
                        .order_by(Project.name).all()

    return render_template(
        'portfolio_results.html',
        portfolio           = pf,
        unassigned_projects = unassigned_projects,
        resources_portfolio = list(res_port),
        resource_schedule   = resource_schedule,
        color_map           = color_map,
        **solution
    )

@app.route('/delete_portfolio/<int:pf_id>', methods=['POST'])
def delete_portfolio(pf_id: int):
    pf = Portfolio.query.get_or_404(pf_id)

    (Project.query.filter_by(portfolio_id=pf_id)
                  .update({"portfolio_id": None}, synchronize_session=False))

    for res in Resource.query.filter_by(portfolio_id=pf_id).all():
        TaskResourceUsage.query.filter_by(resource_id=res.id).delete(
            synchronize_session=False)
        db.session.delete(res)

    db.session.delete(pf)
    db.session.commit()

    return redirect(url_for('portfolios'))


@app.route('/portfolio/<int:pf_id>/add_resource', methods=['POST'])
def add_resource_to_portfolio(pf_id):
    name = request.form.get('resource_name', '').strip()
    cap  = request.form.get('resource_capacity', '1').strip()

    if not name or not cap.isdigit():
        return redirect(url_for('view_portfolio', pf_id=pf_id))

    db.session.add(Resource(
        name=name,
        capacity=int(cap),
        portfolio_id=pf_id
    ))
    db.session.commit()
    return redirect(url_for('view_portfolio', pf_id=pf_id))

@app.route('/portfolio/<int:pf_id>/delete_resource/<int:res_id>', methods=['POST'])
def delete_portfolio_resource(pf_id, res_id):
    res = Resource.query.get_or_404(res_id)

    if res.portfolio_id != pf_id:
        return redirect(url_for('view_portfolio', pf_id=pf_id))

    TaskResourceUsage.query.filter_by(resource_id=res_id) \
                           .delete(synchronize_session=False)

    db.session.delete(res)
    db.session.commit()

    return redirect(url_for('view_portfolio', pf_id=pf_id))

@app.route('/portfolio/<int:pf_id>/add_project', methods=['POST'])
def add_project_to_portfolio(pf_id):
    pf = Portfolio.query.get_or_404(pf_id)
    ids = [int(x) for x in request.form.getlist('project_ids')]

    compatible_ids = {p.id for p in compatible_project_query(pf_id)
                                       .filter(Project.id.in_(ids))}
    if not compatible_ids:
        return redirect(url_for('view_portfolio', pf_id=pf_id))

    (Project.query.filter(Project.id.in_(compatible_ids))
                  .update({"portfolio_id": pf_id}, synchronize_session=False))
    db.session.commit()
    return redirect(url_for('view_portfolio', pf_id=pf_id))

@app.route('/portfolio/<int:pf_id>/remove_project/<int:pr_id>', methods=['POST'])
def remove_project_from_portfolio(pf_id, pr_id):
    pf = Portfolio.query.get_or_404(pf_id)
    pr = Project.query.get_or_404(pr_id)

    if pr.portfolio_id == pf.id:
        pr.portfolio_id = None
        db.session.commit()

    return redirect(url_for('view_portfolio', pf_id=pf_id))

def build_resource_schedule(schedule, resources):
    res_name = {r.id: r.name for r in resources}

    out = []
    for entry in schedule:
        for u in (TaskResourceUsage.query
                    .filter_by(task_id=entry["task"])
                    .filter(TaskResourceUsage.amount > 0)):
            out.append({
                "resource_id":   u.resource_id,
                "resource_name": res_name.get(u.resource_id, f"res#{u.resource_id}"),
                "task_name":     entry["name"],
                "amount":        u.amount,
                "start":         entry["start"],
                "end":           entry["end"]
            })
    return out

@app.route('/new_project', methods=['GET', 'POST'])
def new_project():

    if request.method == 'POST':
        pf_raw = request.form.get('portfolio_id', '').strip()
        portfolio_id = int(pf_raw) if pf_raw else None

        project_name_new = request.form.get('project_name_new', '').strip()
        if not project_name_new:
            project_name_new = request.form.get('project_name', '').strip()
        project_selection = request.form.get('project_selection', '').strip()

        if project_name_new:
            project = Project(name=project_name_new,
                              portfolio_id=portfolio_id)
            db.session.add(project)
            db.session.commit()
        else:
            try:
                project = Project.query.get(int(project_selection))
            except (TypeError, ValueError):
                return "Неверный выбор проекта", 400
            if not project:
                return "Проект не найден", 404

            project.portfolio_id = portfolio_id
            db.session.commit()

        resource_names = request.form.getlist('resource_name[]')
        resource_caps  = request.form.getlist('resource_capacity[]')
        resources = []
        for rn, rc in zip(resource_names, resource_caps):
            rn = rn.strip()
            if rn:
                r = Resource(name=rn,
                             capacity=int(rc or 1),
                             project_id=project.id)
                db.session.add(r)
                resources.append(r)
        db.session.commit()

        task_names     = request.form.getlist('task_name[]')
        task_durations = request.form.getlist('task_duration[]')
        tasks = []
        for tn, td in zip(task_names, task_durations):
            tn = tn.strip()
            if tn:
                t = Task(project_id=project.id,
                         name=tn,
                         duration=int(td or 1))
                db.session.add(t)
                tasks.append(t)
        db.session.commit()

        usage_tasks     = request.form.getlist('usage_task[]')
        usage_resources = request.form.getlist('usage_resource[]')
        usage_amounts   = request.form.getlist('usage_amount[]')

        given_pairs = set()

        for ut, ur, ua in zip(usage_tasks, usage_resources, usage_amounts):
            try:
                task_idx  = int(ut)
                res_id_in = int(ur)
                amount    = int(ua)
            except ValueError:
                continue

            if res_id_in <= 0:
                res_idx = abs(res_id_in) - 1
                if 0 <= res_idx < len(resources):
                    res_id_in = resources[res_idx].id
                else:
                    continue

            if 0 < task_idx <= len(tasks):
                db.session.add(TaskResourceUsage(
                    task_id=tasks[task_idx - 1].id,
                    resource_id=res_id_in,
                    amount=amount))
        db.session.commit()

        for idx, t in enumerate(tasks, start=1):
            for r in resources:
                if (idx, r.id) not in given_pairs:
                    db.session.add(TaskResourceUsage(
                        task_id=t.id, resource_id=r.id, amount=0))
        db.session.commit()

        preds = request.form.getlist('predecessor_id[]')
        succs = request.form.getlist('successor_id[]')
        for p_val, s_val in zip(preds, succs):
            try:
                pred_idx = int(p_val); succ_idx = int(s_val)
            except ValueError:
                continue
            if 1 <= pred_idx <= len(tasks) and 1 <= succ_idx <= len(tasks):
                db.session.add(TaskPrecedence(
                    predecessor_id=tasks[pred_idx - 1].id,
                    successor_id  =tasks[succ_idx - 1].id))
        db.session.commit()

        solution_data = optimize_project(project.id)

        db.session.commit()

        return redirect(url_for('view_project', project_id=project.id))

    portfolios = Portfolio.query.all()

    pf_resources = {
        pf.id: [
            {"id": r.id, "name": r.name, "capacity": r.capacity}
            for r in Resource.query.filter_by(portfolio_id=pf.id).all()
        ]
        for pf in portfolios
    }

    return render_template(
        'all_in_one.html',
        mode='input',
        projects   = Project.query.all(),
        resources  = Resource.query.all(),
        portfolios = portfolios,
        pf_resources = pf_resources
    )

@app.route('/projects')
def projects():
    all_projects = Project.query.all()
    return render_template('projects.html', projects=all_projects)

@app.route('/delete_project/<int:project_id>', methods=['POST'])
def delete_project(project_id):
    project = Project.query.get_or_404(project_id)

    for task in project.tasks:
        TaskResourceUsage.query.filter_by(task_id=task.id).delete()
        TaskPrecedence.query.filter(
            (TaskPrecedence.predecessor_id == task.id) |
            (TaskPrecedence.successor_id == task.id)
        ).delete()
        db.session.delete(task)

    for res in project.resources:
        TaskResourceUsage.query.filter_by(resource_id=res.id).delete()
        db.session.delete(res)

    db.session.delete(project)
    db.session.commit()

    return redirect(url_for('projects'))

@app.route('/view_project/<int:project_id>')
def view_project(project_id):
    project = Project.query.get_or_404(project_id)

    solution_data = optimize_project(project.id)

    res_local = Resource.query.filter_by(project_id=project_id).all()

    res_pf = []
    if project.portfolio_id:
        res_pf = Resource.query.filter_by(
            portfolio_id=project.portfolio_id).all()

    resources_all = res_local + res_pf

    resource_schedule = build_resource_schedule(
        solution_data["schedule"], resources_all)

    return render_template('all_in_one.html',
                           mode='results',
                           project=project,
                           resources=res_local,
                           tasks=project.tasks,
                           schedule=solution_data['schedule'],
                           makespan=solution_data['makespan'],
                           status=solution_data['status'],
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
    project = Project.query.get_or_404(project_id)

    if request.method == 'POST':
        project.name = request.form.get('project_name', '').strip()

        pf_raw = request.form.get('portfolio_id', '').strip()
        project.portfolio_id = int(pf_raw) if pf_raw else None

        existing_res = {r.id: r for r in Resource.query
                                         .filter_by(project_id=project.id)}
        f_res_ids   = request.form.getlist('resource_id[]')
        f_res_names = request.form.getlist('resource_name[]')
        f_res_caps  = request.form.getlist('resource_capacity[]')

        for rid, nm, cap in zip(f_res_ids, f_res_names, f_res_caps):
            nm = nm.strip()
            if not nm:
                continue
            cap_int = int(cap or 1)
            if rid:
                res = existing_res.pop(int(rid), None)
                if res:
                    res.name, res.capacity = nm, cap_int
            else:
                db.session.add(Resource(name=nm,
                                        capacity=cap_int,
                                        project_id=project.id))

        for r in existing_res.values():
            TaskResourceUsage.query.filter_by(resource_id=r.id).delete(synchronize_session=False)
            db.session.delete(r)
        db.session.flush()

        existing_tasks = {t.id: t for t in project.tasks}
        f_task_ids   = request.form.getlist('task_id[]')
        f_task_names = request.form.getlist('task_name[]')
        f_task_durs  = request.form.getlist('task_duration[]')

        idx_map: dict[int, Task] = {}
        for idx, (tid, nm, du) in enumerate(zip(f_task_ids,
                                                f_task_names,
                                                f_task_durs), start=1):
            nm = nm.strip()
            if not nm:
                continue
            du_int = int(du or 1)
            if tid:
                task = existing_tasks.pop(int(tid), None)
                if task:
                    task.name, task.duration = nm, du_int
            else:
                task = Task(project_id=project.id, name=nm, duration=du_int)
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

        for row_idx, res_id_raw, amt_raw in zip(
                request.form.getlist('usage_task[]'),
                request.form.getlist('usage_resource[]'),
                request.form.getlist('usage_amount[]')):

            try:
                task_obj   = idx_map.get(int(row_idx))
                res_id_in  = int(res_id_raw)
                amount_val = int(amt_raw)
            except ValueError:
                continue

            if res_id_in <= 0:
                continue

            if task_obj:
                db.session.add(TaskResourceUsage(
                    task_id    = task_obj.id,
                    resource_id= res_id_in,
                    amount     = amount_val))

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
                        predecessor_id=p_task.id,
                        successor_id=s_task.id))
            except ValueError:
                continue

        db.session.commit()

        solution_data   = optimize_project(project.id)

        local_resources = Resource.query.filter_by(project_id=project.id).all()
        portfolio_resources = []
        if project.portfolio_id:
            portfolio_resources = Resource.query.filter_by(
                portfolio_id=project.portfolio_id).all()

        resources_for_sched = local_resources + portfolio_resources

        resource_schedule = build_resource_schedule(
            solution_data["schedule"], resources_for_sched)
        
        usage_data: dict[tuple[int, int], int] = {}
        for u in TaskResourceUsage.query.filter(
                TaskResourceUsage.task_id.in_([t.id for t in idx_map.values()])):
            key = (u.task_id, u.resource_id)
            usage_data[key] = max(usage_data.get(key, 0), u.amount)

        resource_schedule = build_resource_schedule(
            solution_data["schedule"], resources_for_sched)

        return render_template('edit_project.html',
                               mode='results',
                               project=project,
                               resources=Resource.query.filter_by(project_id=project.id).all(),
                               tasks=list(idx_map.values()),
                               usage_data=usage_data,
                               solution=solution_data,
                               resource_schedule=resource_schedule,
                               portfolios=Portfolio.query.all(),
                               pf_resources = {})

    resources         = Resource.query.filter_by(project_id=project.id).all()
    tasks             = project.tasks
    task_index_map    = {t.id: i for i, t in enumerate(tasks, 1)}

    usage_data = {
        (u.task_id, u.resource_id): u.amount
        for u in TaskResourceUsage.query
                .join(Task)
                .filter(Task.project_id == project.id)
    }

    precedences = TaskPrecedence.query.filter(
        TaskPrecedence.predecessor_id.in_([t.id for t in tasks]),
        TaskPrecedence.successor_id.in_([t.id for t in tasks])
    ).all()

    local_resources = resources
    portfolio_resources = Resource.query.filter_by(
        portfolio_id=project.portfolio_id).all() if project.portfolio_id else []

    resources_for_sched = local_resources + portfolio_resources
    full_resources = local_resources + portfolio_resources

    all_portfolios = Portfolio.query.all()

    pf_resources = {
        pf.id: [
            {"id": r.id, "name": r.name, "capacity": r.capacity}
            for r in Resource.query.filter_by(portfolio_id=pf.id).all()
        ]
        for pf in all_portfolios
    }

    return render_template(
        'edit_project.html',
        mode='input',
        project=project,
        resources=local_resources,    
        full_resources=full_resources,
        tasks=tasks,
        usage_data=usage_data,
        precedences=precedences,
        task_index_map=task_index_map,
        resource_schedule=build_resource_schedule([], resources_for_sched),
        portfolios=all_portfolios,
        pf_resources=pf_resources
    )


def optimize_project(project_id: int):
    project = Project.query.get(project_id)
    if not project:
        return {"status": "error", "msg": "Project not found"}

    tasks = project.tasks

    local_res      = Resource.query.filter_by(project_id=project_id)
    portfolio_res  = Resource.query.filter_by(
                         portfolio_id=project.portfolio_id) \
                     if project.portfolio_id else []

    resources = list(local_res) + list(portfolio_res)

    precedences = (TaskPrecedence.query
                   .join(Task, TaskPrecedence.successor_id == Task.id)
                   .filter(Task.project_id == project_id).all())

    task_ids   = [t.id for t in tasks]
    durations  = {t.id: t.duration for t in tasks}
    resource_caps = {r.id: r.capacity for r in resources}

    usage_map = {}
    for t in tasks:
        for u in t.resource_usages:
            usage_map[(t.id, u.resource_id)] = u.amount

    precedence_list = [(p.predecessor_id, p.successor_id) for p in precedences]
    task_names      = {t.id: t.name for t in tasks}

    return run_optimization_multi(
        task_ids, durations, resource_caps,
        usage_map, precedence_list, task_names)

def optimize_portfolio(portfolio_id: int):
    pf = Portfolio.query.get(portfolio_id)
    if not pf:
        return {"status": "error", "msg": "Portfolio not found"}

    tasks = [t for p in pf.projects for t in p.tasks]
    task_ids  = [t.id for t in tasks]
    durations = {t.id: t.duration for t in tasks}
    task_names = {t.id: f"{t.project.name}: {t.name}" for t in tasks}

    res_port = list(Resource.query.filter_by(portfolio_id=portfolio_id))
    res_proj = list(Resource.query.filter(Resource.project_id.in_([p.id for p in pf.projects])))
    resources = res_port + res_proj
    resource_caps = {r.id: r.capacity for r in resources}

    usage_map = {}
    for t in tasks:
        for u in t.resource_usages:
            usage_map[(t.id, u.resource_id)] = u.amount

    precedence_list = []
    for p in pf.projects:
        precedence_list += [(tp.predecessor_id, tp.successor_id)
                            for tp in TaskPrecedence.query
                                .join(Task, TaskPrecedence.successor_id == Task.id)
                                .filter(Task.project_id == p.id)]

    return run_optimization_multi(task_ids, durations, resource_caps,
                                  usage_map, precedence_list, task_names)

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