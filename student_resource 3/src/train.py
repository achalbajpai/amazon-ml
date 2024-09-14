   import torch
   from torch.utils.data import DataLoader
   from model import setup_model
   from prepare_data import train_dataset

   def train():
       model, processor, device = setup_model()
       train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

       optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

       for epoch in range(10):  # Adjust number of epochs as needed
           for img_paths, labels in train_loader:
               pixel_values = processor(images=[Image.open(path).convert("RGB") for path in img_paths], return_tensors="pt").pixel_values.to(device)
               
               # Prepare labels
               task_prompt = "<s_cord-v2>"
               label_sequences = [f"{task_prompt} {label}" for label in labels]
               label_encoding = processor.tokenizer(label_sequences, padding="max_length", max_length=512, return_tensors="pt")
               
               outputs = model(pixel_values=pixel_values, labels=label_encoding.input_ids)
               loss = outputs.loss
               
               loss.backward()
               optimizer.step()
               optimizer.zero_grad()

           print(f"Epoch {epoch+1} completed")

       model.save_pretrained("./fine_tuned_model")
       processor.save_pretrained("./fine_tuned_model")

   if __name__ == "__main__":
       train()