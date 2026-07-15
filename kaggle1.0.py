import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_list_files('shivamb/netflix-shows', path='.in')

kaggle.api.dataset_download_file('shivamb/netflix-shows',path='.in')

kaggle.api.dataset_metadata('shivamb/netflix-shows', path='.in')

dataset = kaggle.api.dataset_view(search='netflix-shows', file_type='csv')
print(dataset)