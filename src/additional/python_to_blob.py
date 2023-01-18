import os
from azure.storage.blob import BlockBlobService





def upload_cloud(local_folder, cloud_folder_name):

    container_name = 'ashtree-archive'+"/"+blob_folder

    for root, dirs, files in os.walk(local_folder, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            common_prefix = os.path.commonprefix([root_path,file_path])
            rel_path = os.path.relpath(file_path,start=common_prefix)

            print(rel_path)

            block_blob_service = BlockBlobService(
                account_name=account_name,
                account_key=account_key
            )


            blob_name = rel_path
            block_blob_service.create_blob_from_path(container_name, blob_name, file_path)

if __name__ == "__main__":
    # defining the local folder to back up to cloud
    root_path = r'\\gb002339ab\IDA_Data\IDA Ash Dieback\Incoming to Upload\Ceredigion 22\GOPRO'
    # defining the name of the folder on the blob storage
    blob_folder = 'Ceredigion 22\GOPRO'
    # credentials to get access to the cloud - Do not modify them
    account_name = 'saue1prdstdlrsashtree01'
    account_key = r'iRVvUg/k+Svh1DlvRHw7Tl7u3mPGfJnA79z1yk/w4RoD1W54u/uk+EfJ21G5sT5K2ernzghiDCns+AStTEMofg=='

    upload_cloud(local_folder=root_path, cloud_folder_name=blob_folder)
