   import pandas as pd
   from src.utils import download_images

   # Load the dataset
   train_df = pd.read_csv('dataset/train.csv')
   test_df = pd.read_csv('dataset/test.csv')

   # Download images
   download_images(train_df['image_url'].tolist(), 'dataset/train_images')
   download_images(test_df['image_url'].tolist(), 'dataset/test_images')

   # Create a dataset class
   class ProductDataset:
       def __init__(self, dataframe, img_dir):
           self.dataframe = dataframe
           self.img_dir = img_dir

       def __len__(self):
           return len(self.dataframe)

       def __getitem__(self, idx):
           img_name = self.dataframe.iloc[idx]['image_url'].split('/')[-1]
           img_path = f"{self.img_dir}/{img_name}"
           label = self.dataframe.iloc[idx]['label']
           return img_path, label

   # Create train and test datasets
   train_dataset = ProductDataset(train_df, 'dataset/train_images')
   test_dataset = ProductDataset(test_df, 'dataset/test_images')