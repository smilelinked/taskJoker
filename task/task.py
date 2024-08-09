import json
import os
import subprocess

from celery import shared_task, current_app
import SimpleITK as sitk
from configs.config import obsConfig
from utils.basic_setting import logger
from AGENT.draw_a_plane import Plane_PoOr, Plane_LMCoLLCoLNC, Distance_Angle
from AGENT.predict_landmarks import predict_npy, load_models, load_environment, predict_now

# 分割和识别关键点所需nii路径
NII_PATH = '/models/ct/nii/ct.nii.gz'
# nii文件本地存储路径
NII_LOCAL_PATH = '/root/ct.nii.gz'
# 分割算法输出DRC路径前缀
DRC_PREFIX = '/models/ct/stl/'
# 识别关键点算法输出路径
INFORMATION_PATH = '/information.json'


def get_file_from_s3(s3_client, bucket_name, object_name):
    resp = s3_client.get_object(Bucket=bucket_name, Key=object_name)
    if resp.get('ResponseMetadata').get('HTTPStatusCode') > 300:
        raise Exception(f"read file {object_name} failed with resp {resp}")
    return resp.get('Body').read()


def upload_file_to_s3(s3_client, bucket_name, object_name, data, content_type=None):
    """
    上传字节流或JSON字典到OBS

    :param s3_client: s3客户端
    :param bucket_name: 目标bucket名称
    :param object_name: 上传到OBS中的文件名或路径
    :param data: 要上传的数据，可以是字节流或JSON字典
    :param content_type: 数据的MIME类型，默认为None，自动推断
    :return: None
    """
    if isinstance(data, dict):
        data = json.dumps(data).encode('utf-8')
        content_type = content_type or 'application/json'
    elif isinstance(data, bytes):
        content_type = content_type or 'application/octet-stream'

    resp = s3_client.put_object(Bucket=bucket_name, Key=object_name, Body=data, ContentType=content_type)
    if resp.get('ResponseMetadata').get('HTTPStatusCode') > 300:
        raise Exception(f"put obj {object_name} failed with resp {resp}")


def upload_files_to_obs(s3_client, bucket_name, file_paths, obs_directory):
    """
    批量上传文件到OBS

    :param s3_client: s3客户端
    :param bucket_name: 目标bucket名称
    :param file_paths: 本地.drc文件路径列表
    :param obs_directory: OBS上的目标目录
    :return: None
    """
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        object_name = os.path.join(obs_directory, file_name)

        with open(file_path, 'rb') as f:
            file_data = f.read()

        # 上传文件到OBS
        upload_file_to_s3(s3_client, bucket_name, object_name, file_data)


@shared_task(bind=True)
def run_nnunet(self, uid, cid):
    # 使用 self.app 来访问 Celery 应用配置
    s3_client = self.app.conf['s3_client']
    bucket_name = self.app.conf['bucket_name']
    obs_prefix = obsConfig['obs_prefix'](uid, cid)
    nii_path = obs_prefix + NII_PATH

    try:
        nii_file = get_file_from_s3(s3_client, bucket_name, nii_path)
        with open(NII_LOCAL_PATH, 'wb') as f:
            f.write(nii_file)
        logger.info(f"read {nii_path} from {obs_prefix} and write to {NII_LOCAL_PATH}")
    except Exception as e:
        logger.error(f"download file {nii_path} from obs failed: {e}")

    # please CHANGE me to real commands or function call.
    # command = [
    #     'nnUNet_predict', '-i', input_path, '-o', '/data', '-t', 'TaskXX', '-m', '3d_fullres'
    # ]
    command = ['sleep', '10']
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(result.stderr)

    drc_path = obs_prefix + DRC_PREFIX
    # TODO
    return drc_path


@shared_task(bind=True)
def run_plane(self, uid, cid):
    s3_client = self.app.conf['s3_client']
    bucket_name = self.app.conf['bucket_name']
    agent_lst = self.app.conf['agent_lst']
    obs_prefix = obsConfig['obs_prefix'](uid, cid)
    nii_path = obs_prefix + NII_PATH

    try:
        nii_file = get_file_from_s3(s3_client, bucket_name, nii_path)
        with open(NII_LOCAL_PATH, 'wb') as f:
            f.write(nii_file)
        logger.info(f"read {nii_path} from {obs_prefix} and write to {NII_LOCAL_PATH}")
    except Exception as e:
        logger.error(f"download file {nii_path} from obs failed: {e}")

    img = sitk.ReadImage(NII_LOCAL_PATH)
    environment = load_environment(input_img=img)
    result = predict_now(agent_lst, environment)

    # ------------- Get Plane ------------------
    a, b, c = Plane_PoOr(img, result, if_save=False)  # 平面方程1
    a1, b1, c1 = Plane_LMCoLLCoLNC(img, result, if_save=False)  # 平面方程2
    angles, distances = Distance_Angle(img, result)  # 角度，距离

    information_path = obs_prefix + INFORMATION_PATH
    # TODO: confirm information content
    information = {
        'RNC': ''
    }

    try:
        upload_file_to_s3(s3_client, bucket_name, information_path, information)
    except Exception as e:
        logger.error(f"upload file {information_path} to obs failed: {e}")

    return information_path
