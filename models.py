from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Portfolio(db.Model):
    __tablename__ = 'portfolios'
    id   = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f"<Portfolio {self.id} {self.name}>"

class Project(db.Model):
    __tablename__ = 'projects'

    id   = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)

    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolios.id'))
    portfolio    = db.relationship('Portfolio', backref='projects')

    resources = db.relationship('Resource', backref='project', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f"<Project {self.id} {self.name}>"

class Task(db.Model):
    __tablename__ = 'tasks'

    id        = db.Column(db.Integer, primary_key=True)
    name      = db.Column(db.String(100), nullable=False)
    duration  = db.Column(db.Integer, nullable=False)

    project_id = db.Column(db.Integer,
                           db.ForeignKey('projects.id'),
                           nullable=False)
    project    = db.relationship('Project', backref='tasks', lazy=True)

    def __repr__(self):
        return f"<Task {self.id} {self.name} dur={self.duration}>"

class Resource(db.Model):
    __tablename__ = 'resources'

    id       = db.Column(db.Integer, primary_key=True)
    name     = db.Column(db.String(100), nullable=False)
    capacity = db.Column(db.Integer, nullable=False, default=1)

    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolios.id'))
    portfolio    = db.relationship('Portfolio', backref='resources')

    project_id   = db.Column(db.Integer, db.ForeignKey('projects.id'))

    def __repr__(self):
        scope = "port" if self.portfolio_id else "proj"
        return f"<Res {self.id} {self.name} cap={self.capacity} {scope}>"

class TaskResourceUsage(db.Model):
    __tablename__ = 'task_resource_usage'

    id          = db.Column(db.Integer, primary_key=True)
    task_id     = db.Column(db.Integer, db.ForeignKey('tasks.id'),     nullable=False)
    resource_id = db.Column(db.Integer, db.ForeignKey('resources.id'), nullable=False)
    amount      = db.Column(db.Integer, nullable=False, default=0)

    task     = db.relationship('Task',     backref='resource_usages', lazy=True)
    resource = db.relationship('Resource', lazy=True)

    def __repr__(self):
        return f"<TRU T={self.task_id} R={self.resource_id} A={self.amount}>"

class TaskPrecedence(db.Model):
    __tablename__ = 'task_precedences'

    id             = db.Column(db.Integer, primary_key=True)
    predecessor_id = db.Column(db.Integer, db.ForeignKey('tasks.id'), nullable=False)
    successor_id   = db.Column(db.Integer, db.ForeignKey('tasks.id'), nullable=False)

    def __repr__(self):
        return f"<TP {self.predecessor_id}->{self.successor_id}>"
