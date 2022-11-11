import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, CanonSmiles
from rdkit import DataStructs

import pymysql
import pandas as pd

if __name__ == "__main__":
    ########################################### 从 chembl 28筛选出PXC大于等于5的分子 ###################################################
    # file path
    path = "data/chembl_28_pxc_5.txt"

    # conncect dba
    db = pymysql.connect(host="localhost", user="root", password="1234", database="chembl_28")

    # set condition
    cursor = db.cursor()
    sql = "SELECT compound_structures.canonical_smiles FROM activities LEFT JOIN compound_structures ON activities.molregno = compound_structures.molregno WHERE activities.standard_type IN ('IC50','EC50','Ki','Kd') AND activities.standard_units = 'nM' AND activities.standard_value <= 10000;"

    # begin search
    try:   
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
        t = []
        for row in results:
            t += row
        t = list(set(t))
        print(len(t))
        with open(path,'w+') as f:
            for l in t:
                if l is not None:
                    f.write(l+"\n")       
    except:
        print("wrong")

    # close dba
    db.close()

    #####################################################################################################################################



  
