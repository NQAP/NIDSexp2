import pandas as pd

hydra_FCA = pd.read_csv('./FCA_DATASET/Test/hydra_FCA.csv')
nikto_FCA = pd.read_csv('./FCA_DATASET/Test/nikto_FCA.csv')
nmap_FCA = pd.read_csv('./FCA_DATASET/Test/nmap_FCA.csv')
sqlmap_FCA = pd.read_csv('./FCA_DATASET/Test/sqlmap_FCA.csv')
xss_FCA = pd.read_csv('./FCA_DATASET/Test/xss_FCA.csv')

hydra_noFCA = pd.read_csv('./FCA_DATASET/Test/hydra_noFCA.csv')
nikto_noFCA = pd.read_csv('./FCA_DATASET/Test/nikto_noFCA.csv')
nmap_noFCA = pd.read_csv('./FCA_DATASET/Test/nmap_noFCA.csv')
sqlmap_noFCA = pd.read_csv('./FCA_DATASET/Test/sqlmap_noFCA.csv')
xss_noFCA = pd.read_csv('./FCA_DATASET/Test/xss_noFCA.csv')

hydra_mapping = {
    0: 'normal',
    1: 'hydra'
}
hydra_FCA['Label'] = hydra_FCA['Label'].replace(hydra_mapping)
hydra_noFCA['Label'] = hydra_noFCA['Label'].replace(hydra_mapping)

nikto_mapping = {
    0: 'normal',
    1: 'nikto'
}
nikto_FCA['Label'] = nikto_FCA['Label'].replace(nikto_mapping)
nikto_noFCA['Label'] = nikto_noFCA['Label'].replace(nikto_mapping)

nmap_mapping = {
    0: 'normal',
    1: 'nmap'
}
nmap_FCA['Label'] = nmap_FCA['Label'].replace(nmap_mapping)
nmap_noFCA['Label'] = nmap_noFCA['Label'].replace(nmap_mapping)

sqlmap_mapping = {
    0: 'normal',
    1: 'sqli'
}
sqlmap_FCA['Label'] = sqlmap_FCA['Label'].replace(sqlmap_mapping)
sqlmap_noFCA['Label'] = sqlmap_noFCA['Label'].replace(sqlmap_mapping)

xss_mapping = {
    0: 'normal',
    1: 'xss'
}
xss_FCA['Label'] = xss_FCA['Label'].replace(xss_mapping)
xss_noFCA['Label'] = xss_noFCA['Label'].replace(xss_mapping)

hydra_FCA.to_csv('./FCA_DATASET/Test/hydra_FCA.csv')
nikto_FCA.to_csv('./FCA_DATASET/Test/nikto_FCA.csv')
nmap_FCA.to_csv('./FCA_DATASET/Test/nmap_FCA.csv')
sqlmap_FCA.to_csv('./FCA_DATASET/Test/sqlmap_FCA.csv')
xss_FCA.to_csv('./FCA_DATASET/Test/xss_FCA.csv')

hydra_noFCA.to_csv('./FCA_DATASET/Test/hydra_noFCA.csv')
nikto_noFCA.to_csv('./FCA_DATASET/Test/nikto_noFCA.csv')
nmap_noFCA.to_csv('./FCA_DATASET/Test/nmap_noFCA.csv')
sqlmap_noFCA.to_csv('./FCA_DATASET/Test/sqlmap_noFCA.csv')
xss_noFCA.to_csv('./FCA_DATASET/Test/xss_noFCA.csv')

