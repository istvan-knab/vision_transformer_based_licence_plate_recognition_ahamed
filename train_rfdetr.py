from ultralytics import RTDETR
import os

def train_rfdetr_gpu():
    model = RTDETR("rtdetr-l.pt")

    results = model.train(
        data="dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=8,        
        device=0,       
        patience=10,      
        name="rfdetr_license_plate_gpu",
        exist_ok=True
    )

    print("\nTraining Finished!")
    print(f"Best model weights saved to: {os.path.join(results.save_dir, 'weights', 'best.pt')}")

if __name__ == "__main__":
    train_rfdetr_gpu()
