import subprocess

from celery import shared_task, current_app
import SimpleITK as sitk
from configs.config import obsConfig
from utils.basic_setting import logger
from AGENT.draw_a_plane import Plane_PoOr, Plane_LMCoLLCoLNC, Distance_Angle
from AGENT.predict_landmarks import predict_npy, load_models,load_environment,predict_now

@shared_task
def upload_file_to_s3(bucket_name, file_name):
    # 从 Celery 应用上下文中获取 S3 客户端
    s3_client = current_app.s3_client
    response = s3_client.get_object(Bucket=bucket_name, Key=file_name)
    file_content = response['Body'].read()
    return file_content


@shared_task(bind=True)
def run_nnunet(self, uid, cid):
    # 使用 self.app 来访问 Celery 应用配置
    s3_client = self.app.conf['s3_client']
    bucket_name = self.app.conf['bucket_name']
    logger.info(f"we get s3 client {s3_client} and bucket name {bucket_name}")
    input_prefix = obsConfig['get_obj_prefix'](uid, cid)
    output_prefix = obsConfig['put_obj_prefix'](uid, cid)

    input_path = f'{input_prefix}/input_image.nii.gz'
    output_path = f'{output_prefix}/output_image.nii.gz'
    logger.info(f" we get input url {input_path} and upload url {output_path}")
    # please CHANGE me to real commands or function call.
    # command = [
    #     'nnUNet_predict', '-i', input_path, '-o', '/data', '-t', 'TaskXX', '-m', '3d_fullres'
    # ]
    command = ['sleep', '10']
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(result.stderr)

    return f'{output_prefix}/output_image.nii.gz'


@shared_task(bind=True)
def run_plane(self, uid, cid):
    s3_client = self.app.conf['s3_client']
    bucket_name = self.app.conf['bucket_name']
    agent_lst = self.app.conf['agent_lst']
    logger.info(f"we get s3 client {s3_client} and bucket name {bucket_name}")
    input_prefix = obsConfig['get_obj_prefix'](uid, cid)
    output_prefix = obsConfig['put_obj_prefix'](uid, cid)

    input_path = f'{input_prefix}/input_image.nii.gz'
    output_path = f'{output_prefix}/output_image.nii.gz'
    logger.info(f" we get input url {input_path} and upload url {output_path}")
    img = sitk.ReadImage('/root/testdata/001.nii.gz')
    environment = load_environment(input_img=img)
    result = predict_now(agent_lst, environment)
    # ------------- Get Plane ------------------
    a, b, c = Plane_PoOr(img, result, if_save=False)  # 平面方程1
    a1, b1, c1 = Plane_LMCoLLCoLNC(img, result, if_save=False)  # 平面方程2
    angles, distences = Distance_Angle(img, result)  # 角度，距离
    logger.info(f'we get plane 1: {a}, {b}, {c}')
    logger.info(f'we get plane 2: {a1}, {b1}, {c1}')
    logger.info(f'we get angele: {angles} and distance: {distences}')
    return f'{output_prefix}/output_image.nii.gz'
