import os
import boto3
import sagemaker

data_dir = 'dataset'
finetuning_data_dir = 'fashion-captioning'
index_data = 'fashion-product-images-dataset.zip'
finetuning_data_files = ['TRAIN_IMAGES.hdf5', 'TRAIN_CAPTIONS.txt', \
                         'TEST_IMAGES.hdf5', 'TEST_CAPTIONS.txt', \
                         'VAL_IMAGES.hdf5', 'VAL_CAPTIONS.txt']

s3 = boto3.resource('s3')
sess = sagemaker.Session()
sagemaker_bucket = sess.default_bucket()

# 업로드 training dataset
for filename in finetuning_data_files:
    data_file = os.path.join(data_dir, finetuning_data_dir, filename)
    s3.meta.client.upload_file(data_file, sagemaker_bucket, 'training-data/'+filename)

# 추후 인덱싱을 위해 업로드
index_file = os.path.join(data_dir, index_data)
s3.meta.client.upload_file(index_file, sagemaker_bucket, 'index-data/'+index_data)
