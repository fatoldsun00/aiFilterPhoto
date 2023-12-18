
import os
import torch

from langchain.document_loaders import ImageCaptionLoader
from transformers import AutoProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

list_image_urls = []
for filename in os.listdir(os.path.abspath(os.path.join('assetsForTest'))):
    list_image_urls.append(os.path.abspath(os.path.join('assetsForTest', filename)))

#model = BlipForConditionalGeneration.from_pretrained("./blip-manu-finetuned")
#model.to(device)

# query blip about image
loader = ImageCaptionLoader(images=list_image_urls, blip_processor="Salesforce/blip-image-captioning-base", blip_model="./blip-manu-finetuned")
documents = loader.load()
#Parcourir et afficher les légendes et les métadonnées
for doc in documents:
   print("Légende:", doc.page_content)

exit()
# load dataset
from datasets import load_dataset 

dataset = load_dataset("imagefolder", data_dir=os.path.abspath(os.path.join('assetsForTrainBlip')), split="train")
print(dataset[0])

from torch.utils.data import Dataset, DataLoader

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        print(item,"-------------------------------------------------------")
        print (item['text'])
    
        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding
    

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

train_dataset = ImageCaptioningDataset(dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.to(device)

model.train()

for epoch in range(2):
  print("Epoch:", epoch)
  for idx, batch in enumerate(train_dataloader):
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)

    outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)
    
    loss = outputs.loss

    print("Loss:", loss.item())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

model.save_pretrained("./blip-manu-finetuned")