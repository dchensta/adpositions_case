import pandas as pd 

def stats(scene_dict, fn_dict, dir) :
    df = pd.read_csv(dir)
    scene_role = df["scene_role"]
    fn_role = df["fn_role"]
    
    #Count number of rows that have entries.
    count = 0
    #Unique scene roles
    for row in scene_role: 
        if type(row) == str:
            count += 1
            if scene_dict.get(row) == None :
                scene_dict[row] = 1
            else :
                scene_dict[row] += 1
    print("Number of annotations: ", count)
    print("\nUnique scene roles: ")
    for key in scene_dict.keys() :
        print(key)

    for fn in fn_role :
        if type(fn) == str :
            if fn_dict.get(fn) == None :
                fn_dict[fn] = 1
            else :
                fn_dict[fn] += 1
    print("\nUnique function roles: ")
    for key in fn_dict.keys() :
        print(key)


if __name__ == "__main__" :
    pp_dir = "full_annotations/pp_4-5.csv"
    r_dir = "full_annotations/regulus_4-5.csv"

    scene_dict = {} ; fn_dict = {}
    print("Running stats for Pikku Prinssi...")
    stats(scene_dict, fn_dict, pp_dir)
    print("\n")
    print("Running stats for Regulus...")
    stats(scene_dict, fn_dict, r_dir)