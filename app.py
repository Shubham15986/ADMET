import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors, AllChem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.ML.Descriptors import MoleculeDescriptors
from chembl_webresource_client.new_client import new_client
import requests
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import warnings
import joblib
import logging
import json
import pubchempy as pcp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import multiprocessing
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=GEMINI_API_KEY)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="experimental_relax_shapes is deprecated")
warnings.filterwarnings("ignore", message="to-Python converter for boost::shared_ptr<RDKit::FilterHierarchyMatcher>")
warnings.filterwarnings("ignore", message="to-Python converter for boost::shared_ptr<RDKit::FilterCatalogEntry>")

app = Flask(__name__)

# SQLite setup
def init_db():
    conn = sqlite3.connect("admet_data.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS compounds 
                 (smiles TEXT PRIMARY KEY, molwt REAL, logp REAL, hdonors INTEGER, hacceptors INTEGER,
                  absorption INTEGER, distribution INTEGER, metabolism INTEGER, excretion INTEGER, toxicity INTEGER,
                  caco2 REAL, bbb REAL, pgp_inhibition INTEGER, pgp_substrate INTEGER, 
                  cyp_inhibition TEXT, herg_blocking INTEGER, ames_toxicity INTEGER, 
                  dili INTEGER, skin_sensitization INTEGER)''')
    conn.commit()
    conn.close()
    os.makedirs('models', exist_ok=True)

# Configure logging with DEBUG level
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Original models dictionary (for ACO optimization)
original_models = {
    "admet": None,
    "caco2": None,
    "bbb": None,
    "pgp": None,
    "cyp": None,
    "toxicity": None,
    "qed_predictor": None
}

# PAINS filter catalog setup
params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
pains_catalog = FilterCatalog(params)

# Expanded SMARTS-based reaction rules for modifications
reactions = [
    (Chem.AllChem.ReactionFromSmarts('[C:1][H]>>[C:1][OH]'), 'add OH'),  # Aliphatic carbon
    (Chem.AllChem.ReactionFromSmarts('[c:1][H]>>[c:1][OH]'), 'add OH aromatic'),  # Aromatic carbon
    (Chem.AllChem.ReactionFromSmarts('[C:1][H]>>[C:1][NH2]'), 'add NH2'),
    (Chem.AllChem.ReactionFromSmarts('[C:1][H]>>[C:1][F]'), 'add F'),
    (Chem.AllChem.ReactionFromSmarts('[C:1][H]>>[C:1][Cl]'), 'add Cl'),
    (Chem.AllChem.ReactionFromSmarts('[C:1][H]>>[C:1]C'), 'add CH3'),  # Methylation
]

# Fitness function for multi-objective optimization
def compute_fitness(qed, logp, toxicity_score):
    normalized_qed = qed
    normalized_logp = max(0, 1 - abs(logp - 2) / 5)
    normalized_toxicity = max(0, 1 - toxicity_score / 3)
    fitness = 0.4 * normalized_qed + 0.3 * normalized_logp + 0.3 * normalized_toxicity
    logger.debug(f"Computed fitness: QED={qed:.3f}, LogP={logp:.3f}, Toxicity={toxicity_score}, Fitness={fitness:.3f}")
    return fitness

# Helper functions
def nitrogen_count(mol):
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)

def has_alerting_groups(mol):
    smarts_alerts = ["[N+](=O)[O-]", "[N-]=[N+]=[N-]", "C(=O)N(C(=O))N"]
    for smarts in smarts_alerts:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            return True
    return False

def has_skin_alerts(mol):
    smarts_alerts = ["c1([F,Cl,Br,I])ccccc1", "C=CC(=O)"]
    for smarts in smarts_alerts:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            return True
    return False

def create_radar_plot(props):
    labels = ['MolWt', 'LogP', 'HDonors', 'HAcceptors', 'TPSA', 'RotBonds']
    values = [props.get(label, 0) for label in labels]
    max_values = [500, 5, 5, 10, 140, 10]
    values = [min(v / m if m != 0 else 0, 1) for v, m in zip(values, max_values)]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def compute_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            logger.warning(f"Invalid SMILES in compute_features: {smiles}")
            return None
        
        mol = Chem.AddHs(mol)
        mol_volume = None
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mol_volume = AllChem.ComputeMolVolume(mol)
        except Exception as e:
            logger.warning(f"Failed to generate 3D conformer for {smiles}: {str(e)}")
        
        props = {
            "MolWt": Descriptors.MolWt(mol),
            "LogP": Crippen.MolLogP(mol),
            "HDonors": Lipinski.NumHDonors(mol),
            "HAcceptors": Lipinski.NumHAcceptors(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "HBonds": Lipinski.NumHDonors(mol) + Lipinski.NumHAcceptors(mol),
            "RingCount": rdMolDescriptors.CalcNumRings(mol),
            "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "HeavyAtoms": mol.GetNumHeavyAtoms(),
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
            "MolecularVolume": mol_volume if mol_volume is not None else "N/A"
        }
        
        logs_esol = -0.89 + 0.0148 * props["HeavyAtoms"] - 0.876 * np.log(props["MolWt"]) - 0.00728 * props["RotBonds"] - 0.0041 * props["AromaticRings"]
        props["LogS"] = logs_esol
        
        lipo = {
            "CrippenLogP": Crippen.MolLogP(mol),
            "WildmanLogP": Crippen.MolLogP(mol) * 0.95 + 0.2,
            "ConsensusLogP": np.mean([Crippen.MolLogP(mol), Crippen.MolLogP(mol) * 0.95 + 0.2])
        }
        
        if logs_esol >= -1:
            sol_class = "Very Soluble"
        elif logs_esol >= -2:
            sol_class = "Soluble"
        elif logs_esol >= -4:
            sol_class = "Moderately Soluble"
        else:
            sol_class = "Poorly Soluble"
        
        lipinski = sum([props["MolWt"] <= 500, props["LogP"] <= 5, props["HDonors"] <= 5, props["HAcceptors"] <= 10])
        lipinski_class = "Yes" if lipinski >= 3 else "No"
        
        ghose = sum([160 <= props["MolWt"] <= 480, -0.4 <= props["LogP"] <= 5.6, 20 <= props["HeavyAtoms"] <= 70, 0 <= props["RotBonds"] <= 15])
        ghose_class = "Yes" if ghose >= 3 else "No"
        
        veber = sum([props["RotBonds"] <= 10, props["TPSA"] <= 140])
        veber_class = "Yes" if veber == 2 else "No"
        
        muegge = sum([200 <= props["MolWt"] <= 600, -2 <= props["LogP"] <= 5, props["TPSA"] <= 150, 
                      props["RingCount"] <= 7, props["HDonors"] <= 5, props["HAcceptors"] <= 10, props["RotBonds"] <= 15])
        muegge_class = "Yes" if muegge >= 6 else "No"
        
        qed_score = QED.qed(mol)
        
        synth_complexity = min(10, 0.5 * props["RotBonds"] + 0.5 * props["RingCount"] + 0.5 * props["AromaticRings"] + 0.2 * props["HeavyAtoms"])
        synth_access = max(1, 10 - synth_complexity)
        
        pains_matches = pains_catalog.GetMatches(mol)
        toxicity_score = len(pains_matches)
        
        pk = {
            "GIAbsorption": "High" if props["TPSA"] < 140 and 2 < props["LogP"] < 6 else "Low",
            "BBBPermeant": "Yes" if props["TPSA"] < 90 and props["LogP"] < 3 else "No",
            "PgpSubstrate": "Yes" if props["MolWt"] > 400 and props["HAcceptors"] > 8 else "No",
            "CYP1A2Inhibitor": "Yes" if props["AromaticRings"] > 1 else "No",
            "CYP2C9Inhibitor": "Yes" if props["LogP"] > 3 else "No",
            "CYP2D6Inhibitor": "Yes" if nitrogen_count(mol) > 1 and props["LogP"] > 3 else "No",
            "CYP3A4Inhibitor": "Yes" if props["MolWt"] > 400 and props["LogP"] > 3 else "No"
        }
        
        radar_img = create_radar_plot(props)
        
        return {
            "Physicochemical": props,
            "Lipophilicity": lipo,
            "Solubility": {"LogS_ESOL": logs_esol, "Classification": sol_class},
            "DrugLikeness": {
                "Lipinski": lipinski_class,
                "Ghose": ghose_class,
                "Veber": veber_class,
                "Muegge": muegge_class,
                "QED": qed_score
            },
            "MedicinalChem": {
                "QED": qed_score,
                "SyntheticAccessibility": synth_access,
                "ToxicityScore": toxicity_score
            },
            "Pharmacokinetics": pk,
            "Radar": radar_img
        }
    except Exception as e:
        logger.error(f"Error in compute_features: {str(e)}")
        return None

class ADMEPredictor:
    def __init__(self, n_jobs=-1):
        if n_jobs == -1:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)
        self.n_jobs = n_jobs
        self.models = {}
        self.scalers = {}
        self.descriptors = None
        self.setup_descriptors()
        self.performance = {}

    def setup_descriptors(self):
        descriptors = [desc[0] for desc in Descriptors._descList]
        self.descriptors = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors)
        self.descriptor_names = descriptors

    def calculate_features(self, smiles_list):
        features = []
        valid_smiles = []
        
        for smiles in tqdm(smiles_list, desc="Calculating molecular features"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                descriptors = list(self.descriptors.CalcDescriptors(mol))
                descriptors.append(QED.qed(mol))
                descriptors.append(Lipinski.NumHDonors(mol))
                descriptors.append(Lipinski.NumHAcceptors(mol))
                descriptors.append(Chem.Crippen.MolLogP(mol))
                descriptors.append(Chem.Descriptors.TPSA(mol))
                descriptors.append(Chem.Lipinski.NumRotatableBonds(mol))
                descriptors.append(mol.GetNumAtoms())
                features.append(descriptors)
                valid_smiles.append(smiles)
        
        feature_names = self.descriptor_names + ['QED', 'HBD', 'HBA', 'LogP', 'TPSA', 'RotBonds', 'NumAtoms']
        features_df = pd.DataFrame(features, columns=feature_names)
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.dropna(axis=1)
        
        corr_matrix = features_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        features_df = features_df.drop(to_drop, axis=1)
        
        return features_df, valid_smiles

    def train(self, data_path, target_properties, categorical_cutoffs=None):
        data = pd.read_csv(data_path)
        features, valid_smiles = self.calculate_features(data['SMILES'].tolist())
        valid_indices = [i for i, smiles in enumerate(data['SMILES'].tolist()) if smiles in valid_smiles]
        filtered_data = data.iloc[valid_indices]
        
        missing_props = [prop for prop in target_properties if prop not in filtered_data.columns]
        if missing_props:
            raise ValueError(f"Missing properties in dataset: {missing_props}")
        
        for prop in target_properties:
            logger.info(f"Training model for {prop}...")
            y = filtered_data[prop].values
            X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[prop] = scaler
            
            if categorical_cutoffs and prop in categorical_cutoffs:
                cutoff = categorical_cutoffs[prop]
                y_train_binary = (y_train > cutoff).astype(int)
                y_test_binary = (y_test > cutoff).astype(int)
                
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                base_model = RandomForestClassifier(random_state=42, n_jobs=self.n_jobs)
                grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=5, scoring='f1', n_jobs=self.n_jobs)
                grid_search.fit(X_train_scaled, y_train_binary)
                best_model = grid_search.best_estimator_
                
                y_pred = best_model.predict(X_test_scaled)
                self.performance[prop] = {
                    'accuracy': accuracy_score(y_test_binary, y_pred),
                    'precision': precision_score(y_test_binary, y_pred),
                    'recall': recall_score(y_test_binary, y_pred),
                    'f1': f1_score(y_test_binary, y_pred),
                    'type': 'classification',
                    'cutoff': cutoff,
                    'best_params': grid_search.best_params_
                }
                logger.info(f"Classification performance for {prop}: {self.performance[prop]}")
                self.models[prop] = best_model
            else:
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'learning_rate': [0.01, 0.1]
                }
                base_model = GradientBoostingRegressor(random_state=42)
                grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=self.n_jobs)
                grid_search.fit(X_train_scaled, y_train)
                best_model = grid_search.best_estimator_
                
                y_pred = best_model.predict(X_test_scaled)
                self.performance[prop] = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'type': 'regression',
                    'best_params': grid_search.best_params_
                }
                logger.info(f"Regression performance for {prop}: {self.performance[prop]}")
                self.models[prop] = best_model

    def predict(self, smiles_list):
        features, valid_smiles = self.calculate_features(smiles_list)
        results = {'SMILES': valid_smiles}
        
        for prop, model in self.models.items():
            scaled_features = self.scalers[prop].transform(features)
            if self.performance[prop]['type'] == 'classification':
                probs = model.predict_proba(scaled_features)[:, 1]
                results[f"{prop}_probability"] = probs
                results[prop] = (probs > 0.5).astype(int)
            else:
                preds = model.predict(scaled_features)
                results[prop] = preds
        
        return pd.DataFrame(results)

    def plot_feature_importance(self, property_name, top_n=20):
        if property_name not in self.models:
            raise ValueError(f"No model available for {property_name}")
        
        model = self.models[property_name]
        feature_names = self.descriptor_names + ['QED', 'HBD', 'HBA', 'LogP', 'TPSA', 'RotBonds', 'NumAtoms']
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Importances for {property_name}')
        plt.bar(range(top_n), importances[indices][:top_n], align='center')
        plt.xticks(range(top_n), [feature_names[i] for i in indices][:top_n], rotation=90)
        plt.tight_layout()
        plt.show()

    def save_models(self, directory):
        os.makedirs(directory, exist_ok=True)
        for prop in self.models:
            joblib.dump(self.models[prop], os.path.join(directory, f"{prop}_model.pkl"))
            joblib.dump(self.scalers[prop], os.path.join(directory, f"{prop}_scaler.pkl"))
        pd.DataFrame(self.performance).to_csv(os.path.join(directory, "performance.csv"))
        logger.info(f"Models saved to {directory}")

    def load_models(self, directory):
        files = os.listdir(directory)
        model_files = [f for f in files if f.endswith("_model.pkl")]
        for model_file in model_files:
            prop = model_file.replace("_model.pkl", "")
            self.models[prop] = joblib.load(os.path.join(directory, model_file))
            scaler_file = f"{prop}_scaler.pkl"
            if scaler_file in files:
                self.scalers[prop] = joblib.load(os.path.join(directory, scaler_file))
        performance_file = os.path.join(directory, "performance.csv")
        if os.path.exists(performance_file):
            perf_df = pd.read_csv(performance_file, index_col=0)
            self.performance = perf_df.to_dict()
        logger.info(f"Loaded {len(self.models)} models from {directory}")

class ADMEBenchmark:
    def __init__(self, adme_predictor):
        self.predictor = adme_predictor

    def benchmark(self, test_data_path, reference_data_path=None):
        test_data = pd.read_csv(test_data_path)
        predictions = self.predictor.predict(test_data['SMILES'].tolist())
        merged_data = pd.merge(test_data, predictions, on='SMILES', how='inner', suffixes=('_exp', '_pred'))
        
        benchmark_results = {}
        for prop in self.predictor.models:
            if f"{prop}_exp" in merged_data.columns:
                y_true = merged_data[f"{prop}_exp"].values
                if self.predictor.performance[prop]['type'] == 'classification':
                    y_pred = merged_data[prop].values
                    cutoff = self.predictor.performance[prop]['cutoff']
                    y_true_binary = (y_true > cutoff).astype(int)
                    metrics = {
                        'accuracy': accuracy_score(y_true_binary, y_pred),
                        'precision': precision_score(y_true_binary, y_pred),
                        'recall': recall_score(y_true_binary, y_pred),
                        'f1': f1_score(y_true_binary, y_pred)
                    }
                else:
                    y_pred = merged_data[prop].values
                    metrics = {
                        'mae': mean_absolute_error(y_true, y_pred),
                        'r2': r2_score(y_true, y_pred)
                    }
                benchmark_results[prop] = metrics
        
        if reference_data_path:
            reference_data = pd.read_csv(reference_data_path)
            comparison_data = pd.merge(merged_data, reference_data, on='SMILES', how='inner')
            reference_tools = [col.split('_')[0] for col in reference_data.columns if col != 'SMILES' and '_' in col]
            
            for tool in reference_tools:
                for prop in self.predictor.models:
                    ref_col = f"{tool}_{prop}"
                    if ref_col in comparison_data.columns and f"{prop}_exp" in comparison_data.columns:
                        y_true = comparison_data[f"{prop}_exp"].values
                        y_tool = comparison_data[ref_col].values
                        if self.predictor.performance[prop]['type'] == 'classification':
                            cutoff = self.predictor.performance[prop]['cutoff']
                            y_true_binary = (y_true > cutoff).astype(int)
                            if np.unique(y_tool).size > 2:
                                y_tool = (y_tool > cutoff).astype(int)
                            tool_metrics = {
                                'accuracy': accuracy_score(y_true_binary, y_tool),
                                'precision': precision_score(y_true_binary, y_tool),
                                'recall': recall_score(y_true_binary, y_tool),
                                'f1': f1_score(y_true_binary, y_tool)
                            }
                        else:
                            tool_metrics = {
                                'mae': mean_absolute_error(y_true, y_tool),
                                'r2': r2_score(y_true, y_tool)
                            }
                        benchmark_results[f"{tool}_{prop}"] = tool_metrics
        
        logger.info("\nBenchmark Results:")
        for prop, metrics in benchmark_results.items():
            logger.info(f"\n{prop}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        return benchmark_results

class AdvancedADMEAnalyzer:
    def __init__(self, adme_predictor):
        self.predictor = adme_predictor

    def analyze_molecule(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES string"}
        
        mol_props = {
            "molecular_weight": Descriptors.MolWt(mol),
            "logP": Crippen.MolLogP(mol),
            "num_hba": Lipinski.NumHAcceptors(mol),
            "num_hbd": Lipinski.NumHDonors(mol),
            "tpsa": Descriptors.TPSA(mol),
            "qed": QED.qed(mol),
            "num_rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "num_rings": Chem.Lipinski.RingCount(mol),
            "num_aromatic_rings": Chem.Lipinski.NumAromaticRings(mol),
            "num_atoms": mol.GetNumAtoms()
        }
        
        lipinski_violations = 0
        if mol_props["molecular_weight"] > 500: lipinski_violations += 1
        if mol_props["logP"] > 5: lipinski_violations += 1
        if mol_props["num_hba"] > 10: lipinski_violations += 1
        if mol_props["num_hbd"] > 5: lipinski_violations += 1
        mol_props["lipinski_violations"] = lipinski_violations
        
        veber_violations = 0
        if mol_props["num_rotatable_bonds"] > 10: veber_violations += 1
        if mol_props["tpsa"] > 140: veber_violations += 1
        mol_props["veber_violations"] = veber_violations
        
        predictions = self.predictor.predict([smiles])
        if len(predictions) == 0:
            return {"error": "Failed to generate predictions", "properties": mol_props}
        
        analysis = {
            "properties": mol_props,
            "predictions": predictions.to_dict('records')[0]
        }
        
        if lipinski_violations <= 1 and veber_violations == 0 and mol_props["qed"] > 0.5:
            druglikeness = "High"
        elif lipinski_violations <= 2 and veber_violations <= 1 and mol_props["qed"] > 0.3:
            druglikeness = "Medium"
        else:
            druglikeness = "Low"
        
        analysis["druglikeness"] = druglikeness
        return analysis

    def batch_analyze(self, smiles_list):
        results = []
        for smiles in tqdm(smiles_list, desc="Analyzing molecules"):
            analysis = self.analyze_molecule(smiles)
            if "error" not in analysis:
                flat_dict = {"SMILES": smiles}
                for key, value in analysis["properties"].items():
                    flat_dict[f"prop_{key}"] = value
                for key, value in analysis["predictions"].items():
                    if key != "SMILES":
                        flat_dict[f"pred_{key}"] = value
                flat_dict["druglikeness"] = analysis["druglikeness"]
                results.append(flat_dict)
        
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()

    def visualize_property_space(self, smiles_list, color_property=None):
        from sklearn.decomposition import PCA
        features, valid_smiles = self.predictor.calculate_features(smiles_list)
        if color_property and color_property in self.predictor.models:
            predictions = self.predictor.predict(valid_smiles)
            color_values = predictions[color_property].values
        else:
            color_values = [Chem.Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in valid_smiles]
            color_property = "logP"
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=color_values, alpha=0.7, s=50, cmap='viridis')
        plt.colorbar(scatter, label=color_property)
        plt.title(f'Molecular Property Space Colored by {color_property}')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.tight_layout()
        plt.show()

# Initialize ADME Predictor
adme_predictor = ADMEPredictor(n_jobs=4)
model_dir = "adme_models"
train_data_path = "adme_training_data.csv"
properties = [
    "Caco2_Perm", "LogS", "BBB", "CYP3A4", "Pgp_inhibition",
    "HIA", "Bioavailability", "Half_Life", "VDss", "Clearance"
]
categorical_cutoffs = {
    "BBB": 0.3, "CYP3A4": 0.5, "Pgp_inhibition": 0.5, "HIA": 0.8
}

if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
    logger.info(f"Loading existing models from {model_dir}...")
    adme_predictor.load_models(model_dir)
elif os.path.exists(train_data_path):
    logger.info(f"Training models using data from {train_data_path}...")
    adme_predictor.train(train_data_path, properties, categorical_cutoffs)
    adme_predictor.save_models(model_dir)
else:
    logger.warning(f"Training data file not found: {train_data_path}. Predictions will use rule-based methods only.")

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'smiles' not in data:
            logger.warning("Invalid or missing input data")
            return jsonify({"error": "Missing 'smiles' in request body"}), 400
        
        input_data = data['smiles'].strip()
        if not input_data:
            logger.warning("Empty input provided")
            return jsonify({"error": "Input cannot be empty"}), 400
        
        mol = Chem.MolFromSmiles(input_data)
        if not mol:
            logger.info(f"Invalid SMILES '{input_data}', attempting name-to-SMILES conversion")
            smiles = name_to_smiles(input_data)
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
            else:
                logger.error(f"Could not convert '{input_data}' to SMILES")
                return jsonify({"error": f"Could not convert '{input_data}' to SMILES. Please check the input."}), 400
        else:
            smiles = input_data
        
        # Use ADMEPredictor for predictions
        predictions = adme_predictor.predict([smiles])
        analyzer = AdvancedADMEAnalyzer(adme_predictor)
        detailed_analysis = analyzer.analyze_molecule(smiles)
        
        # Original rule-based ADMET
        original_features = compute_features(smiles)
        rules_admet = rules_based_admet(mol)
        
        # ACO optimization
        logger.debug(f"Starting ACO optimization for SMILES: {smiles}")
        optimized_smiles = optimize_smiles_with_aco(smiles, iterations=50, ants=20)
        logger.debug(f"ACO result: Original={smiles}, Optimized={optimized_smiles}")
        optimized_features = compute_features(optimized_smiles) if optimized_smiles else original_features
        
        # Construct response without AI optimization
        response = {
            "input": smiles,
            "predictions": predictions.to_dict('records')[0],
            "detailed_analysis": detailed_analysis,
            "original": {"smiles": smiles, "features": original_features, "radar": original_features.get("Radar", "")},
            "optimized": {"smiles": optimized_smiles, "features": optimized_features, "radar": optimized_features.get("Radar", "")},
            "rules_admet": rules_admet,
        }
        logger.info(f"Successfully processed request for input: {input_data}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

@app.route('/optimize_with_gemini', methods=['POST'])
def optimize_with_gemini():
    try:
        data = request.get_json()
        if not data or 'smiles' not in data:
            logger.warning("Invalid or missing input data")
            return jsonify({"error": "Missing 'smiles' in request body"}), 400
        
        input_data = data['smiles'].strip()
        if not input_data:
            logger.warning("Empty input provided")
            return jsonify({"error": "Input cannot be empty"}), 400
        
        mol = Chem.MolFromSmiles(input_data)
        if not mol:
            logger.info(f"Invalid SMILES '{input_data}', attempting name-to-SMILES conversion")
            smiles = name_to_smiles(input_data)
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
            else:
                logger.error(f"Could not convert '{input_data}' to SMILES")
                return jsonify({"error": f"Could not convert '{input_data}' to SMILES. Please check the input."}), 400
        else:
            smiles = input_data
        
        # Original features for comparison
        original_features = compute_features(smiles)
        orig_qed = original_features["MedicinalChem"]["QED"]
        orig_logp = original_features["Physicochemical"]["LogP"]
        orig_toxicity = original_features["MedicinalChem"]["ToxicityScore"]
        orig_fitness = compute_fitness(orig_qed, orig_logp, orig_toxicity)
        
        # Gemini optimization
        logger.debug(f"Starting Gemini optimization for SMILES: {smiles}")
        gemini_smiles = ai_optimize_smiles(smiles)
        logger.debug(f"Gemini result: Input={smiles}, Optimized={gemini_smiles}")
        gemini_features = compute_features(gemini_smiles) if gemini_smiles else original_features
        
        # Calculate improvements
        gemini_qed = gemini_features["MedicinalChem"]["QED"]
        gemini_logp = gemini_features["Physicochemical"]["LogP"]
        gemini_toxicity = gemini_features["MedicinalChem"]["ToxicityScore"]
        gemini_fitness = compute_fitness(gemini_qed, gemini_logp, gemini_toxicity)
        
        improvements = {
            "QED": gemini_qed - orig_qed,
            "LogP": gemini_logp - orig_logp,
            "Toxicity": orig_toxicity - gemini_toxicity,  # Positive means reduction in toxicity
            "Fitness": gemini_fitness - orig_fitness
        }
        
        response = {
            "input": smiles,
            "original": {"smiles": smiles, "features": original_features, "radar": original_features.get("Radar", "")},
            "gemini_optimized": {"smiles": gemini_smiles, "features": gemini_features, "radar": gemini_features.get("Radar", "")},
            "improvements": improvements
        }
        logger.info(f"Successfully optimized with Gemini for input: {input_data}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Gemini optimization error: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred during Gemini optimization: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            logger.warning("Invalid or missing query data")
            return jsonify({"error": "Missing 'query' in request body"}), 400
        
        query = data['query'].strip()
        if not query:
            logger.warning("Empty query provided")
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # Use Gemini for chat response
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(query)
        response_text = response.text.strip()
        
        logger.debug(f"Chat query: {query}, Response: {response_text}")
        return jsonify({"response": response_text})
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred during chat: {str(e)}"}), 500

# Original Helper Functions
def name_to_smiles(name):
    try:
        compound = pcp.get_compounds(name, 'name')[0]
        return compound.isomeric_smiles
    except Exception as e:
        logger.error(f"Failed to convert {name} to SMILES: {str(e)}")
        return None

def pre_train_ai_model():
    global original_models
    cache_file = "chembl_cache.json"
    compounds = []

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                content = f.read().strip()
                if content:
                    compounds = json.loads(content)
                    logger.info("Loaded ChEMBL data from cache.")
                else:
                    logger.warning("Cache file 'chembl_cache.json' is empty. Fetching new data.")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in cache file '{cache_file}': {str(e)}. Fetching new data.")
        except Exception as e:
            logger.error(f"Error reading cache file '{cache_file}': {str(e)}. Fetching new data.")

    if not compounds:
        try:
            logger.info("Fetching ChEMBL data...")
            molecule = new_client.molecule
            compounds = molecule.filter(molecular_weight__gte=100, molecular_weight__lte=500)[:1000]
            compounds_list = list(compounds)
            with open(cache_file, 'w') as f:
                json.dump(compounds_list, f)
            logger.info("ChEMBL data fetched and cached.")
            compounds = compounds_list
        except Exception as e:
            logger.error(f"Error fetching ChEMBL data: {e}")
            return
    
    features = []
    fitness_scores = []
    for compound in compounds:
        if compound.get('molecule_structures') and 'canonical_smiles' in compound['molecule_structures']:
            smiles = compound['molecule_structures']['canonical_smiles']
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                feat = [
                    Descriptors.MolWt(mol),
                    Crippen.MolLogP(mol),
                    rdMolDescriptors.CalcTPSA(mol),
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol),
                    rdMolDescriptors.CalcNumRotatableBonds(mol)
                ]
                qed = QED.qed(mol)
                logp = Crippen.MolLogP(mol)
                toxicity_score = len(pains_catalog.GetMatches(mol))
                fitness = compute_fitness(qed, logp, toxicity_score)
                features.append(feat)
                fitness_scores.append(fitness)
            else:
                logger.debug(f"Invalid SMILES for compound: {compound.get('molecule_chembl_id', 'unknown')}")
    
    if len(features) >= 2:
        original_models["qed_predictor"] = RandomForestRegressor(n_estimators=100, random_state=42)
        original_models["qed_predictor"].fit(features, fitness_scores)
        logger.info(f"Pre-trained AI fitness predictor with {len(features)} ChEMBL compounds")
    else:
        logger.warning("Not enough valid compounds to train the model")

def rules_based_admet(mol):
    props = compute_features(Chem.MolToSmiles(mol))["Physicochemical"]
    admet = {
        "Absorption": {"prediction": "Good" if props["LogP"] < 5 and props["MolWt"] < 500 else "Poor", "confidence": 0.8},
        "Distribution": {"prediction": "Good" if props["LogP"] > 0 else "Poor", "confidence": 0.7},
        "Metabolism": {"prediction": "Stable" if props["RotBonds"] < 5 else "Unstable", "confidence": 0.6},
        "Excretion": {"prediction": "Fast" if props["MolWt"] < 300 else "Slow", "confidence": 0.7},
        "Toxicity": {"prediction": "Low" if not has_alerting_groups(mol) else "High", "confidence": 0.9}
    }
    return admet

def optimize_smiles_with_aco(smiles, iterations=50, ants=20):
    global ant_database, reactions, pains_catalog
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logger.warning(f"Invalid SMILES for ACO optimization: {smiles}")
        return smiles
    
    try:
        best_smiles = smiles
        best_fitness = compute_fitness(QED.qed(mol), Crippen.MolLogP(mol), len(pains_catalog.GetMatches(mol)))
        pheromone = [1.0] * len(reactions)
        logger.debug(f"Initial SMILES: {smiles}, Fitness: {best_fitness:.3f}")
        
        for iteration in range(iterations):
            ant_solutions = []
            for ant in range(ants):
                total_pheromone = sum(pheromone)
                probabilities = [p / total_pheromone for p in pheromone] if total_pheromone > 0 else [1.0 / len(reactions)] * len(reactions)
                reaction_index = np.random.choice(len(reactions), p=probabilities)
                reaction, reaction_name = reactions[reaction_index]
                smarts_pattern = reaction_name.split()[1] + "[H]"  # Extract the SMARTS pattern from the reaction name
                
                current_mol = Chem.MolFromSmiles(best_smiles)
                if not current_mol:
                    logger.debug(f"Invalid current SMILES in iteration {iteration}, ant {ant}")
                    continue
                    
                matches = current_mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_pattern))
                logger.debug(f"Iteration {iteration}, Ant {ant}: Found {len(matches)} matches for {reaction_name}")
                if not matches:
                    continue
                
                match_idx = np.random.randint(0, len(matches))
                try:
                    products = reaction.RunReactants((current_mol,))
                    logger.debug(f"Reaction {reaction_name} produced {len(products)} product sets")
                    if not products or len(products) == 0 or len(products[0]) == 0:
                        logger.debug(f"No valid products for {reaction_name}")
                        continue
                    
                    new_mol = products[0][0]
                    Chem.SanitizeMol(new_mol)
                    new_smiles = Chem.MolToSmiles(new_mol, canonical=False)  # Avoid canonicalization for testing
                    logger.debug(f"Generated SMILES: {new_smiles}")
                    
                    if new_smiles == best_smiles:
                        logger.debug(f"New SMILES same as best: {new_smiles}")
                        continue
                    
                    feat = compute_features(new_smiles)
                    if not feat:
                        logger.debug(f"Failed to compute features for {new_smiles}")
                        continue
                    
                    qed = feat["MedicinalChem"]["QED"]
                    logp = feat["Physicochemical"]["LogP"]
                    toxicity_score = feat["MedicinalChem"]["ToxicityScore"]
                    fitness = compute_fitness(qed, logp, toxicity_score)
                    
                    logger.debug(f"New SMILES: {new_smiles}, Fitness={fitness:.3f}")
                    ant_solutions.append((reaction_index, fitness, new_smiles))
                    if fitness > best_fitness - 0.01:  # Accept slightly worse for testing
                        best_fitness = fitness
                        best_smiles = new_smiles
                        logger.debug(f"Updated best: {best_smiles}, Fitness={best_fitness:.3f}")
                        break
                
                except Exception as e:
                    logger.debug(f"Error in reaction {reaction_name}: {str(e)}")
                    continue
            
            if ant_solutions:
                for reaction_idx, fitness, _ in ant_solutions:
                    pheromone[reaction_idx] += fitness
                for i in range(len(pheromone)):
                    pheromone[i] *= 0.95
    
        logger.info(f"ACO-Optimized SMILES: {best_smiles} with fitness: {best_fitness:.3f}")
        return best_smiles
    except Exception as e:
        logger.error(f"Error in ACO optimization: {str(e)}", exc_info=True)
        return smiles

def ai_optimize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            logger.warning(f"Invalid SMILES for Gemini optimization: {smiles}")
            return smiles
        
        # Use Gemini AI to suggest an optimized SMILES
        prompt = f"""
        Given the SMILES string '{smiles}' representing a drug molecule, suggest an optimized version of this molecule with improved drug-like properties. 
        Aim to enhance QED (Quantitative Estimate of Drug-likeness), maintain LogP between 1 and 3, and minimize toxicity. 
        Provide the optimized SMILES string in the format: [SMILES: <optimized_smiles>].
        Ensure the new SMILES is a valid chemical structure different from the original.
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        logger.debug(f"Gemini response: {response_text}")

        # Extract SMILES from response
        import re
        match = re.search(r'\[SMILES: (.+?)\]', response_text)
        if not match:
            logger.warning("Gemini did not return a valid SMILES string")
            return smiles
        
        new_smiles = match.group(1)
        new_mol = Chem.MolFromSmiles(new_smiles)
        if not new_mol or new_smiles == smiles:
            logger.warning(f"Gemini returned invalid or unchanged SMILES: {new_smiles}")
            return smiles
        
        # Canonicalize the new SMILES for consistency
        new_smiles = Chem.MolToSmiles(new_mol, canonical=True)
        logger.info(f"Gemini-Optimized SMILES: {new_smiles}")
        return new_smiles
    except Exception as e:
        logger.error(f"Error in Gemini AI optimization: {str(e)}", exc_info=True)
        return smiles

if __name__ == '__main__':
    init_db()
    app.run(debug=True)