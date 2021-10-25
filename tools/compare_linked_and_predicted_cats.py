import numpy as np
from collections.abc import Iterable
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="""This script performs part of radio component
association. Given the catalogue produced by PyBDSF after augmenting the cutout
(Generated by testing_lowflux.py), and the predicted catalogue from a trained fast-rcnn,
compares those two catalogues.""")
parser.add_argument('-d','--debug', help='Enabling debug will render debug output and plots.',
        dest='debug', action='store_true', default=False)
parser.add_argument('-a','--augmented-cat-path', required=True, help='Path to augmented catalogue \
        (the catalogue that was created by running pybdsf over the augmented cutout',
        dest='augmented_cat_path')
parser.add_argument('-l','--linked-cat-path', required=True, help='Path to linked catalogue',
        dest='linked_cat_path')
parser.add_argument('-p','--predicted-cat-path', required=True, help='Path to predicted catalogue',
        dest='predicted_cat_path')
args = vars(parser.parse_args())

debug = args['debug']
linked_cat_path = args['linked_cat_path']
augmented_cat_path = args['augmented_cat_path']
predicted_cat_path = args['predicted_cat_path']

def debug_print(*text):
    if debug:
        print(*text)
# Open catalogues
augmented_cat = pd.read_hdf(augmented_cat_path)
linked_cat = pd.read_hdf(linked_cat_path)
sorted_linked_cat = linked_cat.sort_values(by='Total_flux', ascending=False)
predicted_cat = pd.read_hdf(predicted_cat_path)
sorted_predicted_cat = predicted_cat.sort_values(by='Total_flux', ascending=False)
debug_print("Agumented cat:")
debug_print(augmented_cat)
debug_print("linked_cat:")
debug_print(sorted_linked_cat)
debug_print("predicted cat:")
debug_print(sorted_predicted_cat)

# Add sources that were not included in linked and predicted cat 
augm_names = augmented_cat.Source_Name.values
augm_index = augmented_cat.Source_Name.index
debug_print("All sourcenames:", augm_names)


def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

# Find all sources in linkedcat and predictcat
link_names = list(flatten(sorted_linked_cat.Source_id.values))
predict_names_unflattened = [row.strip('[').strip(']').replace("'","").split() 
        for row in sorted_predicted_cat.Source_Name.values]
predict_names = list(flatten(predict_names_unflattened))
#print('linknames', link_names, type(link_names[0]))
#print('predictnames',predict_names, type(predict_names[0]))
#print('augcat', augmented_cat)
name_to_linkedflux = {str(n):f for n,f in zip(linked_cat.Source_id.values,linked_cat.Total_flux.values)}
name_to_flux ={str(n):f for n,f in zip(augmented_cat.Source_Name.values,augmented_cat.Total_flux.values)}
name_to_predict={}
for row in predicted_cat.Source_Name.values:
    stripped_row = row.strip('[').strip(']').replace("'","").split()
    for s in stripped_row:
        name_to_predict[s] = stripped_row
debug_print("name_to_predict:", name_to_predict)

# Iterate over source_ids
#print("sorted linked cat:", sorted_linked_cat)
#print("sorted predicted cat:", sorted_predicted_cat)
#print("Iterating over linked_cat:")
multi_comp_total_fluxes = [] #[ln for ln in linknames if not ln in predict_names]
single_comp_total_fluxes = [] #[ln for ln in linknames if not ln in predict_names]
component_fractions_correctly_predicted = []
flux_fractions_correctly_predicted = []
flux_fractions_overestimated = []
flux_fractions_missing = []
tallied_comps = []
number_of_linked_components = [len(row) if isinstance(row, np.ndarray) else 1 for row in sorted_linked_cat.Source_id.values]
for row in sorted_linked_cat.Source_id.values:
    print("\nrow:", row, type(row))
    if isinstance(row, np.ndarray):
        row_total_flux = name_to_linkedflux[str(row)]
        row_fractions = []
        row_overestimated=[]
        overestimated_flux = 0
        missing_flux = 0
        for v in row:
            # Check if source is already tallied 
            if str(v) in tallied_comps:
                continue
            else:
                # Check if source is present in predicted cat at all
                if not str(v) in predict_names:
                    print(v, "not present in RCNN predicted catalogue")
                    tallied_comps.append(str(v))
                    missing_flux += name_to_flux[str(v)]
                    continue
                # Check if source is present in predicted cat
                else: 
                    #print("v is in predictnames:", v)
                    predicted = name_to_predict[str(v)]
                    #print("predicted =",predicted)
                    subtotal_flux = 0
                    link_flag = False
                    for p in predicted:
                        #print("p, type, row, type", p, type(p), row, type(row))
                        if p in tallied_comps:
                            continue
                        if int(p) in row:
                            link_flag=True
                            tallied_comps.append(p)
                            subtotal_flux += name_to_flux[p]
                        else:
                            if link_flag:
                                overestimated_flux += name_to_flux[p]
                            tallied_comps.append(p)
                    row_fractions.append(subtotal_flux/row_total_flux)
        flux_fractions_correctly_predicted.append(row_fractions)
        multi_comp_total_fluxes.append(row_total_flux)
        flux_fractions_overestimated.append(overestimated_flux/row_total_flux)
        flux_fractions_missing.append(missing_flux/row_total_flux)
    else:
        single_comp_total_fluxes.append(name_to_linkedflux[str(row)])



# Report stats
debug_print('\nTallied comps:', tallied_comps)
print("\nWe recovered the following stats:" )
print("Flux fractions correctly predicted:")
print(flux_fractions_correctly_predicted)
print("Flux fractions overestimated:")
print(flux_fractions_overestimated)
print("Flux fractions missing:")
print(flux_fractions_missing)
