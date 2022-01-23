import os
import csv
import re

gb_dir = 'gradient_boosting/'
gb_filename = 'gradient_boosting_test_results.csv'
gb_header = ['Accuracy', 'learning_rate', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'max_depth']
gb_re_dict = {
    'Accuracy':re.compile('Accuracy=0\.\d+'),
    'learning_rate':re.compile('learning_rate=\d+\.\d+'),
    'n_estimators':re.compile('n_estimators=\d+'),
    'min_samples_split':re.compile('min_samples_split=\d+'),
    'min_samples_leaf':re.compile('min_samples_leaf=\d+'),
    'max_depth':re.compile('max_depth=\d+')
}
gb_default_data = {
    'Accuracy':0.0,
    'learning_rate':0.1,
    'n_estimators':100,
    'min_samples_split':2,
    'min_samples_leaf':1,
    'max_depth':3
}

rf_dir = 'random_forest/'
rf_filename = 'random_forest_test_results.csv'
rf_header = ['Accuracy', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'max_depth']
rf_re_dict = {
    'Accuracy':re.compile('Accuracy=0\.\d+'),
    'n_estimators':re.compile('n_estimators=\d+'),
    'min_samples_split':re.compile('min_samples_split=\d+'),
    'min_samples_leaf':re.compile('min_samples_leaf=\d+'),
    'max_depth':re.compile('max_depth=\d+')
}
rf_default_data = {
    'Accuracy':0.0,
    'n_estimators':100,
    'min_samples_split':2,
    'min_samples_leaf':1,
    'max_depth':'None'
}

def refactor_procs(dir, filename, header, re_dict, default_data):
    protocols = os.listdir(dir)
    csv_file = open(filename, 'w', newline='')
    writer = csv.writer(csv_file, delimiter=',')
    re_float_int = re.compile('(\d+(?:\.\d+)?)')
    writer.writerow(header)
    for p in protocols:
        protocol_file = open(dir+p)
        data = default_data
        for line in protocol_file:
            for key,value in re_dict.items():
                match = value.search(line)
                if match != None:
                    m_value = re_float_int.search(match.group())
                    data[key] = m_value.group()
        protocol_file.close()
        writer.writerow(data.values())
    csv_file.close()

refactor_procs(dir=gb_dir, filename=gb_filename, header=gb_header, re_dict=gb_re_dict, default_data=gb_default_data)
refactor_procs(dir=rf_dir, filename=rf_filename, header=rf_header, re_dict=rf_re_dict, default_data=rf_default_data)
