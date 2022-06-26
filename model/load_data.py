MESSAGE_ERROR_API_CREDENTIALS = 'Create API credentials!\n\
1)On your Kaggle account, under API, select Create New API Token.\
kaggle.json will be downloaded on your computer.\n\
2)Go to directory — C:\\Users\\{username}\\.kaggle\\ — \
and paste here downloaded JSON file.'


def upload_dataset(dataset_name_='puneet6060/intel-image-classification', path_='./model/content/', unzip_=True):
    try:
        import kaggle
        kaggle.api.authenticate()
    except Exception as e:
        print(e)
        print(MESSAGE_ERROR_API_CREDENTIALS)
        return -1

    kaggle.api.dataset_download_files(dataset_name_, path=path_, unzip=unzip_)
    return 0


# For test
if __name__ == "__main__":
    upload_dataset()
