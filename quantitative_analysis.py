import os
import re
import xlsxwriter as xw


def model_test(dataset, mining_constrain, model_names, relation_patterns):
    '''
    Attributes:
        dataset          :  FB15K237 or WordNet
        
        mining_constrain :  minhc_0.5_minpca_0.9_maxad_4 or
                            minhc_0.5_minpca_0.8_maxad_4 or
                            minhc_0.3_minpca_0.6_maxad_4 or
                            minhc_0.1_minpca_0.4_maxad_4 or
                            minhc_0.1_minpca_0.2_maxad_4
        
        model_names      :  "TransE", "RotatE", "HAKE", "ComplEx", "DualE", "PairRE", and "DistMult" are available
        
        relation_patterns:  "symmetric","inverse","subrelation", and "compose2" are available
    '''
    database_path = os.path.join("dataset",dataset)
    rule_mining_path = os.path.join(database_path,"relation_classify",mining_constrain) #读文件路径
    target_path = os.path.join( rule_mining_path, "test_result") #存文件路径
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    num_contrain = [0,5,10,15,20]
    # model_names = ["TransE", "RotatE", "HAKE", "ComplEx", "DualE", "PairRE", "DistMult"]

    relation_patterns = [ 
                "symmetric",\
                "inverse",\
                "subrelation",\
                "compose2",\
              ]
    
    pre_path = "FreeBase" if dataset == "FB15K237" else "WordNet"
    suf_path = "_FB.sh" if dataset == "FB15K237" else "_WN.sh"
    model_paths = []
    for model_name in model_names:
        model_paths.append(os.path.join(pre_path,model_name+suf_path))

    for index, model in enumerate(model_paths):
        model_name = model_names[index]
        current_target_path = os.path.join( target_path, model_name + ".txt")

        os.system(" echo \"" + model_name + "\" > " + current_target_path)
        # print("=====================baseline=====================")
        # os.system("cat " + database_path + "/test_copy.txt > " + database_path + "/test.txt")
        # os.system("sh scripts/" + model)

        for relation_pattern in relation_patterns:
            for num in num_contrain: 
                print("=====================" + model_name + relation_pattern + " num_contrain " + str(num) +"=====================")
                os.system("echo \"=====================\"" + relation_pattern + "\" num_contrain \"" + str(num) + "\"=====================\" >> " + current_target_path)
                test_file = os.path.join( rule_mining_path, relation_pattern, "num_constrain_"+ str(num) + ".txt" )
                count = len(open(test_file,'r').readlines())
                if count == 0:
                    os.system("echo \" the file is empty, skip this test \" >> " + current_target_path)
                    os.system("echo \"'Test|mrr': 0.0 \" >> " + current_target_path)
                    continue
                
                change_file_order = "cat " + test_file + " > " + database_path + "/test.txt"
                print(change_file_order)
                os.system(change_file_order)

                sh_order = "sh scripts/" + model + " >> " + current_target_path
                print(sh_order)
                os.system(sh_order)

                    
def xw_toExcel(data, fileName, num_contrain):  # xlsxwriter库储存数据到excel
    workbook = xw.Workbook(fileName)                # Create workbook
    worksheet1 = workbook.add_worksheet("sheet1")   # Create worksheet
    worksheet1.activate()                           # Activate worksheet
    title = ['Relation Pattern Dataset']            
    for enm in num_contrain:                        # Set the title of worksheet
        title.append(str(enm))
    print(num_contrain)
    worksheet1.write_row('A1', title)               # Write to the table header starting in cell A1
    i = 2                                           # Write data from the second line
    for j in data:
        print(j)
        row = 'A' + str(i)
        worksheet1.write_row(row, j)
        i += 1
        if (i-1)%(7) == 0:
            row = 'A' + str(i)
            worksheet1.write_row(row, [])
            i += 1
    workbook.close()    # Close the workbook

def creat_test_result_excel(dataset, mining_constrain, model_names, relational_pattern, num_contrain,fileName):
    model_names = ["PairRE", "DistMult"]
    database_path = os.path.join("dataset",dataset)
    rule_mining_path = os.path.join(database_path,"relation_classify",mining_constrain) # Read file path
    target_path = os.path.join( rule_mining_path, "test_result") # Save file path

    result = []
    for model in model_names:
        model_list = []
        model_list.append(str(model))
        result.append(model_list)
        path = os.path.join(target_path,model+".txt")
        # print(path)
        lines = open(path, "r").readlines()
        tmp_result = []
        index = 0
        for line in lines:
            if tmp_result == []:
                tmp_result.append(relational_pattern[index%len(relational_pattern)])
                index += 1
            if re.findall(r"'Test\|mrr': ",line):
                x = re.findall(r"(\d+\.\d{1,4})",line)
                tmp_result.append(x[-1])
            if len(tmp_result)==len(num_contrain) + 1:                              # length of num_contrain
                result.append(tmp_result)
                print(result)
                tmp_result = []
    xw_toExcel(result, fileName, num_contrain)

def main():
    dataset = "FB15K237"                                    # FB15K237 and WN18RR are available
    
    mining_constrain = "minhc_0.5_minpca_0.8_maxad_4"       # select the rule mining threshold, be consistent with those used in the classification

    model_names = [
                    "TransE",\
                    "RotatE", \
                    "HAKE", \
                    "ComplEx", \
                    "DualE", \
                    "PairRE", \
                    "DistMult"
                    ]                                       # select KGE models which are trained before

    relation_patterns = [
                        "symmetric",\
                        "inverse",\
                        "subrelation",\
                        "compose2",\
                        ]                                   # select relational patterns

    model_test(dataset,mining_constrain,model_names, relation_patterns) # quantitative analysis
    
    creat_test_result_excel( dataset, \
                            mining_constrain, \
                            model_names, \
                            relation_patterns, \
                            num_contrain = [0,5,10,15,20], \
                            fileName='statistics_result.xlsx')          # statistics result with excel

if __name__ == "__main__":
    main()