import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors, AllChem

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("test.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Reaction rules
reactions = [
    (Chem.AllChem.ReactionFromSmarts('[C:1][H]>>[C:1][OH]'), 'add OH'),
    (Chem.AllChem.ReactionFromSmarts('[c:1][H]>>[c:1][OH]'), 'add OH aromatic'),
    (Chem.AllChem.ReactionFromSmarts('[C:1][H]>>[C:1][NH2]'), 'add NH2'),
    (Chem.AllChem.ReactionFromSmarts('[C:1][H]>>[C:1][F]'), 'add F'),
    (Chem.AllChem.ReactionFromSmarts('[C:1][H]>>[C:1][Cl]'), 'add Cl'),
    (Chem.AllChem.ReactionFromSmarts('[C:1][H]>>[C:1]C'), 'add CH3'),
]

pains_catalog = FilterCatalog(FilterCatalog(FilterCatalogParams().AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)))

def compute_fitness(qed, logp, toxicity_score):
    normalized_qed = qed
    normalized_logp = max(0, 1 - abs(logp - 2) / 5)
    normalized_toxicity = max(0, 1 - toxicity_score / 3)
    fitness = 0.4 * normalized_qed + 0.3 * normalized_logp + 0.3 * normalized_toxicity
    logger.debug(f"Computed fitness: QED={qed:.3f}, LogP={logp:.3f}, Toxicity={toxicity_score}, Fitness={fitness:.3f}")
    return fitness

def compute_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logger.warning(f"Invalid SMILES in compute_features: {smiles}")
        return None
    props = {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "HDonors": Lipinski.NumHDonors(mol),
        "HAcceptors": Lipinski.NumHAcceptors(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }
    return {
        "Physicochemical": props,
        "MedicinalChem": {"QED": QED.qed(mol), "ToxicityScore": len(pains_catalog.GetMatches(mol))}
    }

def optimize_smiles_with_aco(smiles, iterations=10, ants=5):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        logger.warning(f"Invalid SMILES: {smiles}")
        return smiles
    
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
            
            current_mol = Chem.MolFromSmiles(best_smiles)
            matches = current_mol.GetSubstructMatches(Chem.MolFromSmarts(reaction.GetReactantTemplate(0).ToSmarts()))
            logger.debug(f"Iteration {iteration}, Ant {ant}: Found {len(matches)} matches for {reaction_name}")
            if not matches:
                continue
            
            match_idx = np.random.randint(0, len(matches))
            try:
                products = reaction.RunReactants((current_mol,))
                if not products or len(products) == 0 or len(products[0]) == 0:
                    logger.debug(f"No valid products for {reaction_name}")
                    continue
                
                new_mol = products[0][0]
                Chem.SanitizeMol(new_mol)
                new_smiles = Chem.MolToSmiles(new_mol)
                
                if new_smiles == best_smiles:
                    logger.debug(f"New SMILES same as best: {new_smiles}")
                    continue
                
                feat = compute_features(new_smiles)
                if not feat:
                    continue
                
                qed = feat["MedicinalChem"]["QED"]
                logp = feat["Physicochemical"]["LogP"]
                toxicity_score = feat["MedicinalChem"]["ToxicityScore"]
                fitness = compute_fitness(qed, logp, toxicity_score)
                
                logger.debug(f"New SMILES: {new_smiles}, Fitness={fitness:.3f}")
                ant_solutions.append((reaction_index, fitness, new_smiles))
                if fitness > best_fitness - 0.01:  # Accept even slightly worse for testing
                    best_fitness = fitness
                    best_smiles = new_smiles
                    logger.debug(f"Updated best: {best_smiles}, Fitness={best_fitness:.3f}")
                    break  # Force early exit to test modification
                
            except Exception as e:
                logger.debug(f"Error in reaction {reaction_name}: {str(e)}")
                continue
        
        if ant_solutions:
            for reaction_idx, fitness, _ in ant_solutions:
                pheromone[reaction_idx] += fitness
            for i in range(len(pheromone)):
                pheromone[i] *= 0.95
    
    logger.info(f"ACO Result: {best_smiles}, Fitness: {best_fitness:.3f}")
    return best_smiles

# Test
smiles = "CCCC"  # Simple butane molecule
logger.info(f"Testing ACO optimization with SMILES: {smiles}")
optimized_smiles = optimize_smiles_with_aco(smiles)
logger.info(f"Original: {smiles}, Optimized: {optimized_smiles}")

if optimized_smiles != smiles:
    orig_feat = compute_features(smiles)
    opt_feat = compute_features(optimized_smiles)
    logger.info(f"Original QED: {orig_feat['MedicinalChem']['QED']:.3f}, LogP: {orig_feat['Physicochemical']['LogP']:.3f}")
    logger.info(f"Optimized QED: {opt_feat['MedicinalChem']['QED']:.3f}, LogP: {opt_feat['Physicochemical']['LogP']:.3f}")
else:
    logger.warning("No optimization occurred")