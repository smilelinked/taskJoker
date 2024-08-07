import os

redisConfig = {
    "namespace": "smilelink",
    "key": "crown",
    "redis_config": {
        "host": os.getenv("REDIS", "127.0.0.1"),
        "port": 6379,
        "socket_timeout": None,
        "socket_keepalive": False,
        "ssl": False,
    },
    "maxsize": 100
}

obsConfig = {
    "bucket": os.getenv("BUCKET", "smilelink"),
    "obs_prefix": lambda uid, cid: f'doctor/{uid}/digitalbow-ct/{cid}',
}
