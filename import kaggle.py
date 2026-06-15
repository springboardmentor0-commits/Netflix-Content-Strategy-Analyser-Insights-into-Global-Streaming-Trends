import kaggle

kaggle.api.authenticate()

print(kaggle.api.dataset_list_files('shivamb/netflix-shows'))

kaggle.api.datas0et_download_files('shivamb/netflix-shows', path='.kaggle', unzip=True)

kaggle.api.dataset_metadata('shivamb/netflix-shows', path='.kaggle')
                            
dataset = kaggle.api.dataset_view('netflix shows')                           
print(dataset)