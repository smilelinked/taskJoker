from celery import Celery

from configs.config import redisConfig

CELERY_BROKER_URL = f'redis://{redisConfig["redis_config"]["host"]}:{redisConfig["redis_config"]["port"]}/0',
CELERY_RESULT_BACKEND = f'redis://{redisConfig["redis_config"]["host"]}:{redisConfig["redis_config"]["port"]}/0'


def make_celery(flask_app):
    # 创建 Celery 应用
    celery = Celery(__name__, backend=CELERY_RESULT_BACKEND, broker=CELERY_BROKER_URL)
    celery.conf.update(flask_app.config)

    # 定义任务上下文类
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with flask_app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
