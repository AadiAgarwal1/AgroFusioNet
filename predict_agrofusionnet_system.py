import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

df = pd.read_csv("data/agri_multimodal_dataset_engineered.csv")

metadata = joblib.load("saved_models/agrofusionnet_metadata.pkl")
scaler = joblib.load("saved_models/agrofusionnet_scaler.pkl")
district_map = pd.read_csv("saved_models/district_mapping.csv")

RAIN_COLS = metadata["RAIN_COLS"]
NDVI_COLS = metadata["NDVI_COLS"]
CLIMATE = metadata["CLIMATE"]
SOIL = metadata["SOIL"]
INTERACTION = metadata["INTERACTION"]
STRESS = metadata["STRESS"]
TEMPORAL = metadata["TEMPORAL"]
ALL_FEATURES = metadata["ALL_FEATURES"]

# ------------------------------------------------
# MODEL
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

        yield_pred = self.yield_head(x)
        stress_pred = self.stress_head(x)

        return yield_pred,stress_pred


# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

num_districts = district_map["district_id"].nunique()

model = AgroFusionNet(num_districts)

checkpoint = torch.load("saved_models/agrofusionnet_model.pt",map_location=DEVICE)

if "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

# ------------------------------------------------
# INPUT HELPER
# ------------------------------------------------

def get_input(prompt,default):

    val=input(f"{prompt} (default {round(default,2)}): ")

    if val.strip()=="":
        return default

    return float(val)

# ------------------------------------------------
# DECISION FUNCTIONS
# ------------------------------------------------

def soil_health_score(N,P,K):
    return round(min((N+P+K)/300,1),2)

def climate_impact(sample):

    dev=abs(sample["rainfall_deviation"])
    return round(max(min(1-dev,1),0),2)

def stress_level(prob):

    if prob>0.75:
        return "High"
    elif prob>0.45:
        return "Moderate"
    return "Low"

def sustainability_index(yield_score,stress_prob,soil_score,climate_score):

    si=(0.4*yield_score + 0.3*(1-stress_prob) + 0.2*soil_score + 0.1*climate_score)

    return round(si,2)

def risk_level(si):

    if si>=0.75:
        return "Low Risk"
    elif si>=0.60:
        return "Moderate Risk"
    elif si>=0.45:
        return "High Risk"
    else:
        return "Severe Risk"

# ------------------------------------------------
# CAUSE ANALYSIS
# ------------------------------------------------

def analyze_causes(sample,N,P,K):

    causes = []

    rainfall_dev = sample["rainfall_deviation"]
    ndvi = sample["ndvi"]

    # Climate checks
    if rainfall_dev < -0.15:
        causes.append("Rainfall deficit detected")

    elif rainfall_dev > 0.35:
        causes.append("Excess rainfall detected")

    # Soil nutrient checks
    if N < 60:
        causes.append("Low nitrogen detected")

    elif N > 120:
        causes.append("Excess nitrogen detected")

    if P < 35:
        causes.append("Low phosphorus detected")

    if K < 40:
        causes.append("Low potassium detected")

    # Vegetation health
    if ndvi < 0.45:
        causes.append("Vegetation stress detected")

    return causes

def recommendations(causes):

    rec = []

    for c in causes:

        if "Rainfall deficit" in c:
            rec.append("Increase irrigation frequency")

        if "Excess rainfall" in c:
            rec.append("Improve drainage management")

        if "Low nitrogen" in c:
            rec.append("Apply nitrogen fertilizer")

        if "Excess nitrogen" in c:
            rec.append("Reduce nitrogen application")

        if "Low phosphorus" in c:
            rec.append("Apply phosphorus fertilizer")

        if "Low potassium" in c:
            rec.append("Apply potassium fertilizer")

        if "Vegetation stress" in c:
            rec.append("Inspect crop for pest or disease stress")

    if not rec:
        rec.append("Current conditions appear optimal")

    return rec


# ------------------------------------------------
# MODEL PREDICTION
# ------------------------------------------------

def model_predict(sample, N, P, K, pH, district_id):

    # --------------------------------
    # Copy sample and insert inputs
    # --------------------------------

    sim = sample.copy()

    sim["n"] = N
    sim["p"] = P
    sim["k"] = K
    sim["ph"] = pH

    # --------------------------------
    # Apply scaler AFTER inserting values
    # --------------------------------

    sim_scaled = sim.copy()
    # convert to float before scaling
    sim_scaled[ALL_FEATURES] = sim_scaled[ALL_FEATURES].astype(float)
    sim_scaled.loc[:,ALL_FEATURES] = scaler.transform(sim_scaled[ALL_FEATURES])

    # --------------------------------
    # Build tensors
    # --------------------------------

    rain = torch.tensor(
        sim_scaled[RAIN_COLS].values.reshape(1, 12, 1),
        dtype=torch.float32
    ).to(DEVICE)

    ndvi = torch.tensor(
        sim_scaled[NDVI_COLS].values.reshape(1, 3, 1),
        dtype=torch.float32
    ).to(DEVICE)

    climate = torch.tensor(
        sim_scaled[CLIMATE].values,
        dtype=torch.float32
    ).to(DEVICE)

    soil = torch.tensor(
        sim_scaled[SOIL].values,
        dtype=torch.float32
    ).to(DEVICE)

    inter = torch.tensor(
        sim_scaled[INTERACTION].values,
        dtype=torch.float32
    ).to(DEVICE)

    stress = torch.tensor(
        sim_scaled[STRESS].values,
        dtype=torch.float32
    ).to(DEVICE)

    temp = torch.tensor(
        sim_scaled[TEMPORAL].values,
        dtype=torch.float32
    ).to(DEVICE)

    d = torch.tensor([district_id]).to(DEVICE)

    # --------------------------------
    # Monte Carlo stabilized prediction
    # --------------------------------

    runs = 10
    yield_preds = []
    stress_probs = []

    with torch.no_grad():

        for _ in range(runs):

            y_pred, s_pred = model(
                rain,
                ndvi,
                climate,
                soil,
                inter,
                stress,
                temp,
                d
            )

            yield_preds.append(y_pred.cpu().numpy()[0][0])

            prob = torch.softmax(s_pred, 1).cpu().numpy()[0][2]
            stress_probs.append(prob)

    yield_pred = float(np.mean(yield_preds))

    # --------------------------------
    # Prevent negative yield
    # --------------------------------

    yield_pred = max(yield_pred, 0)

    stress_prob = float(np.mean(stress_probs))

    return yield_pred, stress_prob

# ------------------------------------------------
# SENSITIVITY ANALYSIS
# ------------------------------------------------

def sensitivity_analysis(sample, N, P, K, pH, district_id):

    print("\n📊 Detailed Yield Sensitivity Analysis\n")

    # base prediction
    base_yield, _ = model_predict(sample, N, P, K, pH, district_id)

    # convert to real yield
    base_real = base_yield * 350

    tests = [
        ("Nitrogen +10%", N * 1.1, P, K),
        ("Rainfall +10%", N, P, K),
        ("NDVI -10%", N, P, K)
    ]

    results = []

    for name, n, p, k in tests:

        sim = sample.copy()

        if "Rainfall" in name:
            sim[RAIN_COLS] *= 1.1

        if "NDVI" in name:
            sim[NDVI_COLS] *= 0.9

        y, _ = model_predict(sim, n, p, k, pH, district_id)

        # convert to real yield
        y_real = y * 350

        change = y_real - base_real

        results.append((name, y_real, change))

        print(f"{name:15} → {round(y_real,2)} t/ha ({round(change,2)})")

    most = max(results, key=lambda x: abs(x[2]))

    print(f"\nMost Influential Factor: {most[0]}")

def top_feature_importance(sample,N,P,K,pH,district_id):

    print("\n📊 Top Factors Affecting Yield\n")

    base,_ = model_predict(sample,N,P,K,pH,district_id)
    base_real = base * 350

    feature_tests = {

        "Nitrogen": ("soil",0),
        "Phosphorus": ("soil",1),
        "Potassium": ("soil",2),
        "Soil pH": ("soil",3),
        "Rainfall": ("rain",None),
        "NDVI": ("ndvi",None)

    }

    effects = []

    for name,(group,idx) in feature_tests.items():

        sim = sample.copy()

        n,p,k = N,P,K
        ph_val = pH

        if name=="Nitrogen":
            n = N * 1.1

        elif name=="Phosphorus":
            p = P * 1.1

        elif name=="Potassium":
            k = K * 1.1

        elif name=="Soil pH":
            ph_val = pH * 1.05

        elif name=="Rainfall":
            sim[RAIN_COLS] *= 1.1

        elif name=="NDVI":
            sim[NDVI_COLS] *= 1.1

        y,_ = model_predict(sim,n,p,k,ph_val,district_id)

        y_real = y * 350

        change = abs(y_real - base_real)

        effects.append((name,change))

    effects.sort(key=lambda x: x[1],reverse=True)

    for i,(name,val) in enumerate(effects[:5],1):

        print(f"{i}. {name}")

# ------------------------------------------------
# OUTPUT ENGINE
# ------------------------------------------------

def print_results(sample,N,P,K,pH,district_id):

    yield_pred,stress_prob=model_predict(sample,N,P,K,pH,district_id)

    yield_real = yield_pred * 350

    trend = yield_real - sample["yield_last_year"].values[0]

    soil_score=soil_health_score(N,P,K)
    climate_score=climate_impact(sample.iloc[0])

    yield_score=yield_pred/df["yield"].max()

    si=sustainability_index(yield_score,stress_prob,soil_score,climate_score)
    risk=risk_level(si)

    causes=analyze_causes(sample.iloc[0],N,P,K)
    recs=recommendations(causes)

    print("\n🌾 AgriResearchAI Decision Intelligence System\n")

    print("📊 MODEL RESULTS\n")

    print(f"Predicted Yield:           {round(yield_real,3)} t/ha")
    print(f"Yield Trend:               {round(trend,3)} t/ha\n")

    print(f"Stress Probability:        {round(stress_prob*100,3)} %")
    print(f"Stress Level:              {stress_level(stress_prob)}\n")

    print(f"Soil Health Score:         {soil_score}")
    print(f"Climate Impact Score:      {climate_score}\n")

    print(f"🌍 Sustainability Index:   {si}")
    print(f"⚠ Risk Level:              {risk}\n")

    print("🔎 Causes")
    if len(causes)==0:
        print("• No major risk factors detected")
    else:
        for c in causes:
            print("•",c)

    print("\n💡 Recommendations")
    for r in recs:
        print("•",r)

    sensitivity_analysis(sample,N,P,K,pH,district_id)

    top_feature_importance(sample,N,P,K,pH,district_id)

# ------------------------------------------------
# MAIN SYSTEM
# ------------------------------------------------

print("\n🌾 AgriResearchAI Decision Intelligence System\n")

state=input("Enter State: ")
district=input("Enter District: ")

filtered=df[
(df["state"].str.lower()==state.lower()) &
(df["district"].str.lower()==district.lower())
]

crops=sorted(filtered["crop"].unique())

print("\n🌾 Available Crops:")

for i,c in enumerate(crops,1):
    print(i,c)

choice=int(input("\nSelect Crop Number: "))

crop=crops[choice-1]

sample=filtered[filtered["crop"]==crop].iloc[[-1]]

district_id=int(sample["district_id"].values[0])

print("\n--- Soil Parameters ---")

N=get_input("Soil Nitrogen (N)",sample["n"].values[0])
P=get_input("Soil Phosphorus (P)",sample["p"].values[0])
K=get_input("Soil Potassium (K)",sample["k"].values[0])
pH=get_input("Soil pH",sample["ph"].values[0])

print("\n--- Climate Inputs ---")

sample["annual_rainfall"]=get_input("Annual Rainfall",sample["annual_rainfall"].values[0])
sample["rainfall_deviation"]=get_input("Rainfall Deviation",sample["rainfall_deviation"].values[0])

print("\n--- Vegetation Inputs ---")

sample["ndvi"]=get_input("NDVI",sample["ndvi"].values[0])
sample["ndvi_lag1"]=get_input("NDVI Lag1",sample["ndvi_lag1"].values[0])
sample["ndvi_lag2"]=get_input("NDVI Lag2",sample["ndvi_lag2"].values[0])

print("\n--- Temporal Inputs ---")

sample["year"]=get_input("Year",sample["year"].values[0])

print_results(sample,N,P,K,pH,district_id)

# ------------------------------------------------
# SIMULATION LOOP
# ------------------------------------------------

while True:

    sim_N = N
    sim_P = P
    sim_K = K
    sim_pH = pH


    print("\nSimulation Menu")

    print("1 Modify Nutrients")
    print("2 Modify Rainfall")
    print("3 Modify NDVI")
    print("4 Modify Year")
    print("5 Modify Multiple Factors")
    print("6 Exit Simulation")

    choice = input("Select Option: ").strip()

    if choice == "":
        continue

    option = int(choice)


    if option==6:
        break

    sim=sample.copy()

    if option==1:

        N=float(input("Enter New Nitrogen: "))
        P=float(input("Enter New Phosphorus: "))
        K=float(input("Enter New Potassium: "))

    if option==2:

        sim[RAIN_COLS]*=float(input("Rainfall multiplier: "))

    if option==3:

        sim[NDVI_COLS]*=float(input("NDVI multiplier: "))

    if option==4:

        sim["year"]=float(input("Enter New Year: "))

    if option==5:

        N=float(input("Nitrogen: "))
        P=float(input("Phosphorus: "))
        K=float(input("Potassium: "))

        sim[RAIN_COLS]*=float(input("Rainfall multiplier: "))
        sim[NDVI_COLS]*=float(input("NDVI multiplier: "))

    print("\n📊 Simulation Result")

    print_results(sim,N,P,K,pH,district_id)
