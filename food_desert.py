import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import resample
import folium
from folium import Choropleth, GeoJsonTooltip
import geopandas as gpd

# Loads data
df = pd.read_csv("data/FoodAccessResearchAtlasData2019.csv")


'''
Target variable
Variable definition: Low income and low access tract measured at 1 mile for urban areas and 10 miles for rural areas 
'''
target_col = "LILATracts_1And10"

'''
Selected features:
- Poverty rate
- Median family income
- Urban tract
- TractUNV: Tract housing units without a vehicle, number
- TractWhite: Tract White population, number
- TractBlack: Tract Black or African American population, number
- TractHispanic: Tract Hispanic or Latino population, number
- TractKids: Tract children age 0-17, number
- TractSeniors: Tract seniors age 65+, number
- lahunvhalfshare: Vehicle access, housing units without and low access at 1/2 mile, share
- lapop1share: Low access, population at 1 mile, share
- lalowi1share: Low access, low-income population at 1 mile, share
- TractAsian: Tract Asian population, number
- TractNHOPI: Tract Native Hawaiian and Other Pacific Islander population, number
- TractAIAN: Tract American Indian and Alaska Native population, number
- TractOMultir: Tract Other/Multiple race population, number
'''
features = [
    "PovertyRate", "MedianFamilyIncome", "Urban", "TractHUNV",
    "TractWhite", "TractBlack", "TractHispanic", "TractKids", "TractSeniors",
    "lahunvhalfshare", "lapop1share", "lalowi1share", "TractAsian", "TractNHOPI",
    "TractAIAN", "TractOMultir"
]

# Drops missing values target and features
df = df[df[target_col].notnull()]
df = df.dropna(subset=features)

# Normalizes features between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])
y = df[target_col].values.reshape(-1, 1)

# Splits the data in a 80-20 split for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Combines X_train and y_train into a datafram for resampling, and flattens
train_df = pd.DataFrame(X_train, columns=features)
train_df["target"] = y_train.flatten()  

'''
Our target variable has a lot more census tracts in one class (0), 
so it was causing the model to overpredict "not in food desert"
'''

# Split the data into two classes: majority and minority
majority = train_df[train_df["target"] == 0]
minority = train_df[train_df["target"] == 1]

# Randomly downsampled the majority class to fit the minority class size
majority_downsampled = resample(
    majority,
    replace=False,
    n_samples=len(minority),
    random_state=42
)

# Combined the new majority and minority class into one df
balanced_df = pd.concat([majority_downsampled, minority]).sample(frac=1, random_state=42)

# Extracts the X and y values
X_train_balanced = balanced_df[features].values
y_train_balanced = balanced_df["target"].values.reshape(-1, 1)

# Defines class for the manual implementation of the MLP (has 2 hidden layers)
class ManualMLP:
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, output_size=1, lr=0.1, lambda_reg=0.0001):
        # Sets the learning rate and L2 Regularization values
        self.lr = lr
        self.lambda_reg = lambda_reg

        # Weight initialization for the ReLU activation
        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size1))

        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))

        self.W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2. / hidden_size2)
        self.b3 = np.zeros((1, output_size))

        # To store the losses to plot later to check for convergence
        self.losses = []

    # Defines the ReLU function which returns either x or 0 (no negatives)
    def relu(self, x):
        return np.maximum(0, x)

    # Defines the ReLU derivative which returns 1 if x>0 or 0
    def relu_derivative(self, x):
        return (x > 0).astype(float)

    # Defines the Sigmoid function where its 1/(1+e^-x)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Computes the binary cross entropy loss between predicted and true labels
    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Defines the forward pass through the network
    def forward_pass(self, X):
        # Hidden layer 1: uses ReLU activation
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)

        # Hidden layer 2: uses ReLU activation
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.relu(self.Z2)

        # Output layer: uses Sigmoid activation
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = self.sigmoid(self.Z3)

        return self.A3

    # Defines the backprop through the network to update weights
    def backward_pass(self, X, y, output):
        # The number of samples
        m = X.shape[0]

        # Computes the output lyaer error --> predicted A3 - true y
        dZ3 = output - y
        # Calculates gradient of loss weights and bias for output layer, including L2 regualrization 
        dW3 = (1 / m) * (self.A2.T @ dZ3) + self.lambda_reg * self.W3
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        # Computes the gradient of loss at activation at layer 2 --> applies the ReLU derivative --> gradient of loss for weights at layer 2
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.relu_derivative(self.Z2)
        dW2 = (1 / m) * (self.A1.T @ dZ2) + self.lambda_reg * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Computes the gradient of loss at activation at layer 1 --> applies the ReLU derivative --> gradient of loss for weights at layer 1
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = (1 / m) * (X.T @ dZ1) + self.lambda_reg * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Updates weights
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # To train the model
    def train(self, X, y, epochs=1000, verbose=True):
        # Again, to store losses to check for convergence later
        self.losses = []

        # To iterate over the number of epochs
        for epoch in range(epochs):
            # Computes predicted outputs from X
            output = self.forward_pass(X)
            # Computes the binary cross entropy loss
            loss = self.binary_cross_entropy(y, output)

            # Computes the L2 regularization term to prevent overfitting
            l2_term = (
                self.lambda_reg / 2
            ) * (np.linalg.norm(self.W1)**2 + np.linalg.norm(self.W2)**2 + np.linalg.norm(self.W3)**2)

            # Calculates and stores the total loss for that epoch
            total_loss = loss + l2_term
            self.losses.append(total_loss)

            # Backpropogation: updates weights
            self.backward_pass(X, y, output)

            # To track the progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

    # Predicts whether or not it is a food desert (0 == it is not or 1 == it is)
    # Returns 1 if it is over the given threshodl
    def predict(self, X, threshold=0.6):
        probs = self.forward_pass(X)
        return (probs > threshold).astype(int)

    # Returns the probabilityt that each input is a food desert
    def predict_proba(self, X):
        return self.forward_pass(X)
 
'''
Initializes the model with:
- an input size equal to the number of features
- two hidden layers with 64 and 32 neurons respectively
- a learning rate set to 0.1
'''
model = ManualMLP(
    input_size=X_train.shape[1],
    hidden_size1=64,
    hidden_size2=32,
    lr=0.1
)

# Trains the model on the balanced training data for 2000 epochs
model.train(X_train_balanced, y_train_balanced, epochs=2000)

# Predicts classes (0 or 1) on the original training and testing sets
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)


# Calculates the accuracy for the training and testing sets
train_acc = np.mean(train_preds == y_train)
test_acc = np.mean(test_preds == y_test)

# Prints the accuracy
print(f"\nTrain Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")

# Prints the confusion matrix --> to see how well the model distinguished between classes
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_preds))

# Prints the classification report --> to see how well the model performed by class
print("\nClassification Report:")
print(classification_report(y_test, test_preds))

# Plots the training loss over the number of epochs to check for convergence
plt.plot(model.losses)
plt.title("MLP Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


# For our plot on how well the model did at classifiying whether or not counties are food deserts compared to the USDA's defintions
# Adds USDA true values and MLP model predictions
df["actual_label"] = df["LILATracts_1And10"]
df["predicted_label"] = model.predict(scaler.transform(df[features]), threshold=0.6).flatten()

# Summarizes predictions at the county level, counting total census tracks in the county with MLP predicted and USDA actual food deserts
county_stats = df.groupby(["State", "County"]).agg(
    total_tracts=("predicted_label", "count"),
    predicted_food_deserts=("predicted_label", "sum"),
    actual_food_deserts=("actual_label", "sum")
).reset_index()

# Calculates the percent of census tracts labeled as food deserts
county_stats["percent_predicted"] = (
    county_stats["predicted_food_deserts"] / county_stats["total_tracts"] * 100
)
county_stats["percent_actual"] = (
    county_stats["actual_food_deserts"] / county_stats["total_tracts"] * 100
)

# To determine whether or not the USDA and the model agree on food desert designations for the county (use 50% as the threshold)
def classify_agreement(row, threshold=50): 
    # True Positive: both the USDA and the model agree that over 50% of the county's tracts are food deserts
    if row["percent_actual"] >= threshold and row["percent_predicted"] >= threshold:
        return "True Positive"
    # True Negative: both the USDA and the model agree that under 50% of the county's tracts are food deserts
    elif row["percent_actual"] < threshold and row["percent_predicted"] < threshold:
        return "True Negative"
    # False Positive: USDA says under 50%, but the model says over 50% --> model has overpredicted the risk
    elif row["percent_actual"] < threshold and row["percent_predicted"] >= threshold:
        return "False Positive"
    # False Negative: USDA says over 50%, but the model says under 50% --> model has underpredicted the risk
    elif row["percent_actual"] >= threshold and row["percent_predicted"] < threshold:
        return "False Negative"
    else:
        return "Other"

# Applies classification
county_stats["agreement_type"] = county_stats.apply(classify_agreement, axis=1)

# Defines colors for the plot for each agreement
color_map = {
    "True Positive": "blue",
    "True Negative": "green",
    "False Positive": "red",
    "False Negative": "orange",
}

# Initializes figure dimensions
plt.figure(figsize=(10, 7))

# Iterates over the different types of agreements possible
for label, group in county_stats.groupby("agreement_type"):
    # Creates a scatterplot for each type with the color points defined above
    plt.scatter(
        group["percent_actual"],
        group["percent_predicted"],
        label=label,
        alpha=0.6,
        s=40,
        color=color_map.get(label, "gray")
    )

# Creates a diagnoal line on the plot that represents what perfect agreement would look like
plt.plot([0, 100], [0, 100], linestyle="--", color="gray")

plt.xlabel("USDA Reported Food Desert Rate by County in %")
plt.ylabel("MLP Predicted Food Desert Rate by County in %")
plt.title("Comparing Model Predictions vs. USDA Food Desert Classifications by County")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# For our interactive map:
# Loads the shapefile containing US county boundaries and metadata into a Geo df (this dataset was given to us by our DS4200 Professor)
gdf = gpd.read_file("data/ne_10m_admin_2_counties.shp")

# Defines the state names for state abbreviations (these were named differently between the two datasets)
state_abbr_to_name = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia"
}

# Defines the "county" names for Alaska regions (these were named differently between the two datasets)
alaska_county_names = {
    "Anchorage": "Anchorage Municipality",
    "North Slope": "North Slope Borough",
    "Yukon-Koyukuk": "Yukon-Koyukuk Census Area",
    "Southeast Fairbanks": "Southeast Fairbanks Census Area",
    "Copper River": "Valdez-Cordova Census Area",
    "Yakutat": "Yakutat City and Borough",
    "Hoonah-Angoon": "Hoonah-Angoon Census Area",
    "Skagway": "Skagway Municipality",
    "Haines": "Haines Borough",
    "Juneau": "Juneau City and Borough",
    "Petersburg": "Petersburg Borough",
    "Wrangell": "Wrangell City and Borough",
    "Ketchikan Gateway": "Ketchikan Gateway Borough",
    "Prince of Wales-Hyder": "Prince of Wales-Hyder Census Area",
    "Kodiak Island": "Kodiak Island Borough",
    "Lake and Peninsula": "Lake and Peninsula Borough",
    "Sitka": "Sitka City and Borough",
    "Dillingham": "Dillingham Census Area",
    "Kenai Peninsula": "Kenai Peninsula Borough",
    "Chugach": "Valdez-Cordova Census Area",
    "Nome": "Nome Census Area",
    "Matanuska-Susitna": "Matanuska-Susitna Borough",
    "Aleutians East": "Aleutians East Borough",
    "Bristol Bay": "Bristol Bay Borough",
    "Bethel": "Bethel Census Area",
    "Northwest Arctic": "Northwest Arctic Borough",
    "Aleutians West": "Aleutians West Census Area",
    "Denali": "Denali Borough",
    "Fairbanks North Star": "Fairbanks North Star Borough",
    "Kusilvak": "Kusilvak Census Area"
}

gdf["State"] = gdf["REGION"].map(state_abbr_to_name)

'''
These are targeted states that we noticed were not matching up between the two datasets because of data inconsistencies...

The states below had Saint versus St. written for county names.
'''
target_states = {"FL", "MO", "LA", "AL", "MD", "MN", "NY", "MI", "WI", "IN", "AR"}

gdf["County"] = gdf.apply(
    lambda row: (
        "District of Columbia"
        if row["REGION"] == "DC" and row["NAME"] == "Washington DC"
        else alaska_county_names[row["NAME"]]
        if row["REGION"] == "AK" and row["NAME"] in alaska_county_names
        else (
            row["NAME"].replace("Saint ", "St. ")
            if row["REGION"] in target_states and row["NAME"].startswith("Saint ")
            else row["NAME"].strip()
        ) + (" Parish" if row["REGION"] == "LA" else " County")
    ),
    axis=1
)

# Creates a merged dataframe 
merged = gdf.merge(
    county_stats[["State", "County", "percent_predicted"]],
    left_on=["State", "County"],
    right_on=["State", "County"],
    how="left"
)

# Creates Folium map already centered on the United States
m = folium.Map(location=[37.8, -96], zoom_start=4, tiles="cartodbpositron")

# Choropleth layer of the map --> to display
Choropleth(
    geo_data=merged,
    data=merged,
    columns=["County", "percent_predicted"],  
    key_on="feature.properties.County",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    nan_fill_color="gray",
    legend_name="Predicted Food Desert Tracts by County in %"
).add_to(m)

# Adds an interactive feature where the user can hover over the visualization and it displays the county, state, and the percent predicted as a food desert
folium.GeoJson(
    merged,
    name="Counties",
    style_function=lambda x: {"fillOpacity": 0, "color": "transparent", "weight": 0},
    tooltip=GeoJsonTooltip(
        fields=["County", "REGION", "percent_predicted"],
        aliases=["County:", "State:", "Food Desert (%):"],
        localize=True
    )
).add_to(m)

# Saves the map in an html format
m.save("predicted_food_desert_map.html")