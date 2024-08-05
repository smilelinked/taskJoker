import signal
import boto3
import threading
from app.app import make_celery
from flask import Flask, request, jsonify
from utils.basic_setting import logger
from task.task import run_nnunet


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        uid = request.json['uid']
        cid = request.json['cid']

        task = run_nnunet.apply_async(args=[uid, cid])
        return jsonify({'task_id': task.id}), 202

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/result/<task_id>', methods=['GET'])
def result(task_id):
    task = run_nnunet.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info),
        }
    return jsonify(response)


def signal_handler():
    logger.info("Caught Interrupt, shutting down Flask application")
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    else:
        logger.error("Server shutdown function not found")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Teeth alignment inference server')
    parser.add_argument('--port', type=int, default=8290, help='Port to listen on')
    parser.add_argument('--obs_access_key', type=str, required=True, help='OBS access key')
    parser.add_argument('--obs_secret_key', type=str, required=True, help='OBS secret key')
    parser.add_argument('--obs_endpoint', type=str, default="https://obs.cn-east-3.myhuaweicloud.com",
                        help='OBS endpoint')
    parser.add_argument('--bucket_name', type=str, default="ct", help='OBS bucket name')
    args = parser.parse_args()

    # 设置中断信号处理器
    signal.signal(signal.SIGINT, signal_handler)

    s3 = boto3.client(
        's3',
        aws_access_key_id=args.obs_access_key,
        aws_secret_access_key=args.obs_secret_key,
        endpoint_url=args.obs_endpoint
    )
    app.config['s3_client'] = s3
    app.config['bucket_name'] = args.bucket_name

    # 创建 Celery 应用
    celery_instance = make_celery(app)

    # 启动 Celery worker 线程
    threading.Thread(target=lambda: celery_instance.worker_main(['worker', '--loglevel=info', "--pool=solo"])).start()

    app.run(host='0.0.0.0', port=args.port)