import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

DATA_PATH = "data/agri_multimodal_dataset_engineered.csv"
TARGET = "yield"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001

os.makedirs("saved_models", exist_ok=True)
os.makedirs("experiments", exist_ok=True)

# ------------------------------------------------
# Fix yield units
# ------------------------------------------------

def fix_yield_units(df):

    df = df.copy()

    yield_cols = [
        "yield",
        "yield_last_year",
        "yield_3yr_avg",
        "yield_variance",
        "yield_stability"
    ]

    for col in yield_cols:

        if col in df.columns and df[col].max() > 100:
            df[col] = df[col] / 1000

    return df

# ------------------------------------------------
# Create stress label
# ------------------------------------------------

def create_stress_label(df):

    df = df.copy()

    q50 = df["aesi"].quantile(0.50)
    q75 = df["aesi"].quantile(0.75)

    df["stress"] = 0
    df.loc[df["aesi"] > q75, "stress"] = 2
    df.loc[(df["aesi"] > q50) & (df["aesi"] <= q75), "stress"] = 1

    return df

# ------------------------------------------------
# Load dataset
# ------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df = df.sort_values(["district","crop","year"])

df = fix_yield_units(df)
df = create_stress_label(df)

df = df.fillna(df.median(numeric_only=True))

print("Dataset shape:", df.shape)

# ------------------------------------------------
# Feature Groups
# ------------------------------------------------

RAIN_COLS = [
"rain_jan","rain_feb","rain_mar","rain_apr","rain_may","rain_jun",
"rain_jul","rain_aug","rain_sep","rain_oct","rain_nov","rain_dec"
]

NDVI_COLS = ["ndvi","ndvi_lag1","ndvi_lag2"]

CLIMATE = [
"annual_rainfall","rainfall_avg","rainfall_deviation",
"rainfall_variability","rainfall_shock",
"kharif_rainfall","rabi_rainfall","zaid_rainfall",
"kharif_ratio","rabi_ratio","zaid_ratio"
]

SOIL = [
"n","p","k","ph",
"nitrogen_surplus","nutrient_balance",
"np_ratio","nk_ratio","pk_ratio","npk_balance"
]

INTERACTION = [
"rainfall_n_interaction",
"rainfall_p_interaction",
"rainfall_k_interaction"
]

STRESS = ["snii","aesi"]

TEMPORAL = [
"yield_last_year",
"yield_3yr_avg",
"yield_variance",
"yield_stability"
]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols = RAIN_COLS + NDVI_COLS + CLIMATE + SOIL + INTERACTION + STRESS + TEMPORAL

df[num_cols] = scaler.fit_transform(df[num_cols])

# ------------------------------------------------
# Train Test Split
# ------------------------------------------------

train_df = df[df.year < 2016]
test_df = df[df.year >= 2016]

num_districts = df["district_id"].nunique()

# ------------------------------------------------
# Dataset
# ------------------------------------------------

class AgriDataset(Dataset):

    def __init__(self,df):

        self.rain = df[RAIN_COLS].values.astype(np.float32)
        self.ndvi = df[NDVI_COLS].values.astype(np.float32)
        self.climate = df[CLIMATE].values.astype(np.float32)

        self.soil = df[SOIL].values.astype(np.float32)
        self.inter = df[INTERACTION].values.astype(np.float32)
        self.stress = df[STRESS].values.astype(np.float32)
        self.temp = df[TEMPORAL].values.astype(np.float32)

        self.district = df["district_id"].values.astype(np.int64)

        self.y = df[TARGET].values.astype(np.float32)
        self.s = df["stress"].values.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):

        rain = self.rain[idx].reshape(12,1)
        ndvi = self.ndvi[idx].reshape(3,1)

        return (
            rain,
            ndvi,
            self.climate[idx],
            self.soil[idx],
            self.inter[idx],
            self.stress[idx],
            self.temp[idx],
            self.district[idx],
            self.y[idx],
            self.s[idx]
        )

train_loader = DataLoader(AgriDataset(train_df),batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(AgriDataset(test_df),batch_size=BATCH_SIZE)

# ------------------------------------------------
# Cross Modal Attention
# ------------------------------------------------

class CrossModalAttention(nn.Module):

    def __init__(self,dim):

        super().__init__()

        self.q = nn.Linear(dim,dim)
        self.k = nn.Linear(dim,dim)
        self.v = nn.Linear(dim,dim)

        self.scale = dim**-0.5

    def forward(self,x):

        Q=self.q(x)
        K=self.k(x)
        V=self.v(x)

        attn=torch.softmax((Q@K.transpose(-2,-1))*self.scale,dim=-1)

        return attn@V + x

# ------------------------------------------------
# Gated Fusion
# ------------------------------------------------

class GatedFusion(nn.Module):

    def __init__(self,dim):

        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim,dim),
            nn.Sigmoid()
        )

    def forward(self,x):

        g=self.gate(x)

        return x*g

# ------------------------------------------------
# AgroFusionNet
# ------------------------------------------------

class AgroFusionNet(nn.Module):

    def __init__(self,num_districts):

        super().__init__()

        self.rain_lstm = nn.LSTM(1,32,batch_first=True)
        self.ndvi_lstm = nn.LSTM(1,16,batch_first=True)

        self.climate = nn.Sequential(
            nn.Linear(len(CLIMATE),32),
            nn.ReLU(),
            nn.Linear(32,16)
        )

        self.soil = nn.Sequential(
            nn.Linear(len(SOIL),64),
            nn.ReLU(),
            nn.Linear(64,32)
        )

        self.inter = nn.Sequential(
            nn.Linear(len(INTERACTION),16),
            nn.ReLU()
        )

        self.stress = nn.Sequential(
            nn.Linear(len(STRESS),16),
            nn.ReLU()
        )

        self.temp = nn.Sequential(
            nn.Linear(len(TEMPORAL),16),
            nn.ReLU()
        )

        self.district_embed = nn.Embedding(num_districts,16)

        fusion_size = 160

        self.cross_attn = CrossModalAttention(fusion_size)
        self.gate = GatedFusion(fusion_size)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_size,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32)
        )

        self.yield_head = nn.Linear(32,1)
        self.stress_head = nn.Linear(32,3)

    def forward(self,rain,ndvi,climate,soil,inter,stress,temp,d):

        rain,_ = self.rain_lstm(rain)
        rain = rain[:,-1,:]

        ndvi,_ = self.ndvi_lstm(ndvi)
        ndvi = ndvi[:,-1,:]

        climate = self.climate(climate)
        soil = self.soil(soil)
        inter = self.inter(inter)
        stress = self.stress(stress)
        temp = self.temp(temp)

        d = self.district_embed(d)

        x = torch.cat([
            rain,
            ndvi,
            climate,
            soil,
            inter,
            stress,
            temp,
            d
        ],dim=1)

        x = self.cross_attn(x.unsqueeze(1)).squeeze(1)

        x = self.gate(x)

        x = self.fusion(x)

        yield_pred = self.yield_head(x).squeeze()
        stress_pred = self.stress_head(x)

        return yield_pred,stress_pred
# ------------------------------------------------
# Training
# ------------------------------------------------

model = AgroFusionNet(num_districts).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),lr=LR)

loss_y = nn.MSELoss()
loss_s = nn.CrossEntropyLoss()

print("Training AgroFusionNet...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for rain,ndvi,climate,soil,inter,stress,temp,d,y,s in train_loader:

        rain,ndvi,climate = rain.to(DEVICE),ndvi.to(DEVICE),climate.to(DEVICE)
        soil,inter,stress,temp = soil.to(DEVICE),inter.to(DEVICE),stress.to(DEVICE),temp.to(DEVICE)
        d = d.to(DEVICE)

        y = y.to(DEVICE)
        s = s.to(DEVICE)

        optimizer.zero_grad()

        yp,sp = model(rain,ndvi,climate,soil,inter,stress,temp,d)

        loss = loss_y(yp,y) + loss_s(sp,s)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch",epoch+1,"Loss:",total_loss)

# ------------------------------------------------
# Evaluation
# ------------------------------------------------

model.eval()

y_true=[]
y_pred=[]
s_true=[]
s_pred=[]

with torch.no_grad():

    for rain,ndvi,climate,soil,inter,stress,temp,d,y,s in test_loader:

        rain,ndvi,climate = rain.to(DEVICE),ndvi.to(DEVICE),climate.to(DEVICE)
        soil,inter,stress,temp = soil.to(DEVICE),inter.to(DEVICE),stress.to(DEVICE),temp.to(DEVICE)
        d = d.to(DEVICE)

        yp,sp = model(rain,ndvi,climate,soil,inter,stress,temp,d)

        y_true.extend(y.numpy())
        y_pred.extend(yp.cpu().numpy())

        s_true.extend(s.numpy())
        s_pred.extend(torch.argmax(sp,1).cpu().numpy())

rmse=np.sqrt(mean_squared_error(y_true,y_pred))
mae=mean_absolute_error(y_true,y_pred)
r2=r2_score(y_true,y_pred)

acc=accuracy_score(s_true,s_pred)
prec=precision_score(s_true,s_pred,average="weighted")
rec=recall_score(s_true,s_pred,average="weighted")
f1=f1_score(s_true,s_pred,average="weighted")

print("\nModel Performance")

print("\nYield Regression")
print("RMSE:",rmse)
print("MAE:",mae)
print("R2:",r2)

print("\nStress Classification")
print("Accuracy:",acc)
print("Precision:",prec)
print("Recall:",rec)
print("F1:",f1)

torch.save(model.state_dict(),"saved_models/agrofusionnet_model.pt")

results=pd.DataFrame([{
"Model":"AgroFusionNet",
"RMSE":rmse,
"MAE":mae,
"R2":r2,
"Accuracy":acc,
"Precision":prec,
"Recall":rec,
"F1":f1
}])

results.to_excel("experiments/agrofusionnet_results.xlsx",index=False)

print("\nTraining Completed Successfully")