from typing import Literal, get_args
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

Element = Literal["fe", "c", "mn", "si", "cr", "ni", "mo", "v", "n", "nb", "co", "w", "al", "ti"]
Property = Literal["yield_strength", "tensile_strength", "elongation"]
ElongationCategory = Literal["weak", "medium", "strong", "NaN"]

ELEMENTS: list[Element] = list(get_args(Element))
PROPERTIES: list[Property] = list(get_args(Property))

ELEMENT_WEIGHT: dict[str, float] = {
    'fe': 55.845,
    'c': 12.011,
    'mn': 54.938,
    'si': 28.086,
    'cr': 51.996,
    'ni': 58.693,
    'mo': 95.94,
    'v': 50.942,
    'n': 14.007,
    'nb': 92.906,
    'co': 58.933,
    'w': 183.84,
    'al': 26.982,
    'ti': 47.867
}

class DataframeWriter:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def atom_to_weight_percent(self, formula: str):
        atom_list = list(formula)

        flip = False
        joined_list = []
        k = 0

        for i in range(1, len(atom_list)):
            if not flip:
                k += 1
                if not atom_list[i].isalpha():
                    flip = True

                    if k == 1:
                        joined_list.append(atom_list[i-1])

                    elif k == 2:
                        joined_list.append(atom_list[i-2] + atom_list[i-1])

                    k = 0

            elif flip:
                k += 1
                if atom_list[i].isalpha():
                    flip = False

                    intermediate = ''
                    for j in range(i - k, i):
                        intermediate += atom_list[j]
                    joined_list.append(intermediate)

                    k = 0

                elif i == len(atom_list) - 1:
                    intermediate = ''
                    for j in range(i - k, i + 1):
                        intermediate += atom_list[j]
                    joined_list.append(intermediate)

        formula_tuple_list: list[tuple[str, float]] = []

        for i in range(0, len(joined_list), 2):
            formula_tuple_list.append(tuple((joined_list[i], float(joined_list[i+1]))))

        formula_dict = dict(formula_tuple_list)
        weight_dict_intermediate: dict[str, float] = {}

        for key in formula_dict:
            weight_dict_intermediate[key] = formula_dict[key] * ELEMENT_WEIGHT[key]


        total_weight = sum(weight_dict_intermediate.values())

        weight_dict = {}
        for key in weight_dict_intermediate:
            weight_dict[key] = round(100 * weight_dict_intermediate[key] / total_weight, 2)

        return weight_dict

    def data_fill(self):
        fe_calc = []
        c_calc = []
        mn_calc = []
        si_calc = []
        cr_calc = []
        ni_calc = []
        mo_calc = []
        v_calc = []
        n_calc = []
        nb_calc = []
        co_calc = []
        w_calc = []
        al_calc = []
        ti_calc = []

        for i in range(0, len(self.data.index)):
            weight_dict: dict = (self.atom_to_weight_percent(self.data['formula'][i]))
            fe_calc.append(weight_dict.get('fe', 0))
            c_calc.append(weight_dict.get('c', 0))
            mn_calc.append(weight_dict.get('mn', 0))
            si_calc.append(weight_dict.get('si', 0))
            cr_calc.append(weight_dict.get('cr', 0))
            ni_calc.append(weight_dict.get('ni', 0))
            mo_calc.append(weight_dict.get('mo', 0))
            v_calc.append(weight_dict.get('v', 0))
            n_calc.append(weight_dict.get('n', 0))
            nb_calc.append(weight_dict.get('nb', 0))
            co_calc.append(weight_dict.get('co', 0))
            w_calc.append(weight_dict.get('w', 0))
            al_calc.append(weight_dict.get('al', 0))
            ti_calc.append(weight_dict.get('ti', 0))

        self.data['fe_calc'] = fe_calc
        self.data['c_calc'] = c_calc
        self.data['mn_calc'] = mn_calc
        self.data['si_calc'] = si_calc
        self.data['cr_calc'] = cr_calc
        self.data['ni_calc'] = ni_calc
        self.data['mo_calc'] = mo_calc
        self.data['v_calc'] = v_calc
        self.data['n_calc'] = n_calc
        self.data['nb_calc'] = nb_calc
        self.data['co_calc'] = co_calc
        self.data['w_calc'] = w_calc
        self.data['al_calc'] = al_calc
        self.data['ti_calc'] = ti_calc

        return self.data
    
    @staticmethod
    def data_shuffler(data: pd.DataFrame, n: int):
        sorted_data = data.sort_values(["fe"], ignore_index=True)
        length_of_segment = int(round(len(data.index) / n))
        shuffled_fe_dict: dict[str, pd.DataFrame] = {}

        for i in range(0, n-1):
            shuffled_fe_dict[f"shuffled_fe_data{i+1}"] = (
                sorted_data
                .iloc[i*length_of_segment:(i+1)*length_of_segment]
                .sample(frac=1)
                .reset_index(drop=True)
            )

        shuffled_fe_dict[f"shuffled_fe_data{n}"] = (
            sorted_data
            .iloc[(n-1)*length_of_segment:len(data.index)]
            .sample(frac=1)
            .reset_index(drop=True)
        )

        fe_sample_data: list[pd.DataFrame] = []
        partial_sample_dict: dict[str, pd.DataFrame] = {}

        for i in range(1, n):
            sample_df = pd.DataFrame([])
            for j in range(1, n+1):
                partial_sample_dict[f"partial_sample{j}"] = (
                    shuffled_fe_dict[f"shuffled_fe_data{j}"]
                    .iloc[0:int(round(length_of_segment / n))]
                )
                sample_df = pd.concat([sample_df, partial_sample_dict[f"partial_sample{j}"]])
                shuffled_fe_dict[f"shuffled_fe_data{j}"] = (
                    shuffled_fe_dict[f"shuffled_fe_data{j}"]
                    .reset_index(drop=True)
                    .drop(np.arange(0, int(round(length_of_segment / n))))
                )
            
            fe_sample_data.append(sample_df.reset_index(drop=True))
        
        sample_df = pd.DataFrame([])

        for i in range(1, n+1):
            sample_df = pd.concat([sample_df, shuffled_fe_dict[f"shuffled_fe_data{i}"]])
        
        fe_sample_data.append(sample_df.reset_index(drop=True))

        return fe_sample_data


def a_calc(data: pd.DataFrame):
    data = data.dropna()
    composition_vectors = data[[element for element in ELEMENTS]].to_numpy()
    properties_vectors = data[[properties for properties in PROPERTIES]].to_numpy()

    model = sklearn.linear_model.LinearRegression(fit_intercept = False)
    model.fit(composition_vectors, properties_vectors)

    A = model.coef_

    return A

def calculate_mape(actual, predicted) -> float:  
    return np.mean(np.abs(( actual - predicted) / actual), axis=0) * 100

def calculate_std(actual: np.ndarray, predicted: np.ndarray, mean: float = 0):
    return np.sqrt(np.sum(
        np.square(actual - predicted - mean), axis=0) / (actual - predicted).shape[0]
    )

def categorise_elongation(elongation: float) -> ElongationCategory:
    if np.isnan(elongation):
        return np.nan
    if elongation < 5:
        return "weak"
    if elongation > 10:
        return "strong"
    return "medium"
        
alloy_properties = pd.read_csv(r"C:\Users\sambi\Programming\alloy-properties-ml\database_steel_properties.csv", skiprows=1)
alloy_properties['formula'] = (alloy_properties['formula'].apply(lambda value: value.lower()))

dataframe_writer = DataframeWriter(alloy_properties)

alloy_properties = dataframe_writer.data_fill()

refined_alloy_properties = pd.DataFrame(
    {
        element: alloy_properties[element]
        .combine_first(alloy_properties[f"{element}_calc"])
        for element in ELEMENTS if element != "fe"
    }
)
refined_alloy_properties["fe"] = alloy_properties["fe_calc"]
refined_alloy_properties["combined_compositions"] = (
    refined_alloy_properties[[element for element in ELEMENTS]]
    .values
    .tolist()
)
refined_alloy_properties["yield_strength"] = alloy_properties["yield strength"]
refined_alloy_properties["tensile_strength"] = alloy_properties["tensile strength"]
refined_alloy_properties["elongation"] = alloy_properties["elongation"]
refined_alloy_properties["combined_properties"] = (
    refined_alloy_properties[[prop for prop in PROPERTIES]]
    .values
    .tolist()
)
refined_alloy_properties["combined_properties"] = (
    refined_alloy_properties["combined_properties"]
    .apply(lambda prop: np.array(prop))
)

test_alloy_properties = refined_alloy_properties

A_learned = a_calc(refined_alloy_properties)

composition_vectors = (
    refined_alloy_properties[[element for element in ELEMENT_WEIGHT.keys()]]
    .to_numpy()
)

refined_alloy_properties["elongation_predicted"] = (
    refined_alloy_properties["combined_compositions"]
    .apply(lambda prop: np.matmul(A_learned, prop)[2])
)
refined_alloy_properties["combined_predicted"] = (
    refined_alloy_properties["combined_compositions"]
    .apply(lambda prop: np.matmul(A_learned, prop))
)

reduced_alloy_properties = refined_alloy_properties.dropna()

properties_array = np.array(
    reduced_alloy_properties["combined_properties"]
    .values
    .tolist()
)
predicted_array = np.array(
    reduced_alloy_properties["combined_predicted"]
    .values
    .tolist()
)

r_squared = list(r2_score(
    properties_array,
    predicted_array,
    multioutput="raw_values",
))

std_combined= calculate_std(properties_array, predicted_array)

mape_combined = calculate_mape(properties_array, predicted_array)

refined_alloy_properties["elongation_catagorised_true"] = (
    refined_alloy_properties["elongation"]
    .apply(categorise_elongation)
)
refined_alloy_properties["elongation_catagorised_predicted"] = (
    refined_alloy_properties["elongation_predicted"]
    .apply(categorise_elongation)
)

count_have_data = 0 
count_not_equal = 0

for i in range(0, len(refined_alloy_properties.index)):
    if refined_alloy_properties.loc[i, "elongation_catagorised_true"] == "NaN":
        continue
    count_have_data += 1
    if (refined_alloy_properties.loc[i, "elongation_catagorised_true"] != 
        refined_alloy_properties.loc[i, "elongation_catagorised_predicted"]
    ):
        count_not_equal +=1

print(f"The R^2 for the entire dataset is {r_squared}")
print(f"The std for the entire dataset is {std_combined}")
print(f"The MAPE for the entire dataset is {mape_combined}")
print(f"The percentage of incorrect assignments of elongation is {count_not_equal / count_have_data}")

#
#The following is to see if there is a correlation between error and and element composition
#

refined_alloy_properties["percent_error_combined"] = abs(
    (refined_alloy_properties["combined_properties"] - 
    refined_alloy_properties["combined_predicted"]) / 
    refined_alloy_properties["combined_properties"]
)

refined_alloy_properties["percent_error_ys"] = (
    refined_alloy_properties["percent_error_combined"]
    .apply(lambda prop: prop[0])
)
refined_alloy_properties["percent_error_ts"] = (
    refined_alloy_properties["percent_error_combined"]
    .apply(lambda prop: prop[1])
)
refined_alloy_properties["percent_error_e"] = (
    refined_alloy_properties["percent_error_combined"]
    .apply(lambda prop: prop[2])
)
refined_alloy_properties = refined_alloy_properties.dropna()

figure, axis = plt.subplots(2, 7)

plots: dict[str, plt.Axes] = {}
k = 0

for i in range(0, len(ELEMENTS)):
    j = i
    if j >= int(len(ELEMENTS) / 2):
        k = 1
        j = i - int(len(ELEMENTS) / 2)
    plots[f"{ELEMENTS[i]}_err"] = sns.kdeplot(
        data=refined_alloy_properties, 
        y=f"{ELEMENTS[i]}", 
        x="percent_error_ys", 
        ax=axis[k,j], 
        fill=True, 
        cmap="rocket_r",
    )
    plots[f"{ELEMENTS[i]}_err"].set_xlim(left=0)
    plots[f"{ELEMENTS[i]}_err"].set_ylim(bottom=0)

plots["fe_err"].set_ylim([60,85])
plots["c_err"].set_ylim(top=0.45)
plots["mn_err"].set_ylim(top=1)
plots["si_err"].set_ylim(top=2.5)
plots["cr_err"].set_ylim(top=22.5)
plots["ni_err"].set_ylim(top=25)
plots["mo_err"].set_ylim(top=8)
plots["v_err"].set_ylim(top=1.4)
plots["n_err"].set_ylim(top=0.055)
plots["nb_err"].set_ylim(top=0.25)
plots["co_err"].set_ylim(top=22.5)
plots["w_err"].set_ylim(top=2.5)
plots["al_err"].set_ylim(top=1.4)
plots["ti_err"].set_ylim(top=2.75)

plt.tight_layout(w_pad=-2.5, h_pad=-1)
plt.show()

figure, axis = plt.subplots(2, 7)
k = 0

for i in range(0, len(ELEMENTS)):
    j = i
    if j >= int(len(ELEMENTS) / 2):
        k = 1
        j = i - int(len(ELEMENTS) / 2)
    plots[f"{ELEMENTS[i]}_err"] = sns.kdeplot(
        data=refined_alloy_properties, 
        y=f"{ELEMENTS[i]}", 
        x="percent_error_ts", 
        ax=axis[k,j], 
        fill=True, 
        cmap="rocket_r",
    )
    plots[f"{ELEMENTS[i]}_err"].set_xlim(left=0)
    plots[f"{ELEMENTS[i]}_err"].set_ylim(bottom=0)


plots["fe_err"].set_ylim([60, 85])
plots["c_err"].set_ylim(top=0.5)
plots["mn_err"].set_ylim(top=1.1)
plots["si_err"].set_ylim(top=2.5)
plots["cr_err"].set_ylim(top=22.5)
plots["ni_err"].set_ylim(top=25)
plots["mo_err"].set_ylim(top=8)
plots["v_err"].set_ylim(top=1.4)
plots["n_err"].set_ylim(top=0.055)
plots["nb_err"].set_ylim(top=0.25)
plots["co_err"].set_ylim(top=22.5)
plots["w_err"].set_ylim(top=2.75)
plots["al_err"].set_ylim(top=1.4)
plots["ti_err"].set_ylim(top=2.75)

plt.tight_layout(w_pad=-2.5, h_pad=-1)
plt.show()

figure, axis = plt.subplots(2, 7)
k = 0

for i in range(0, len(ELEMENTS)):
    j = i
    if j >= int(len(ELEMENTS) / 2):
        k = 1
        j = i - int(len(ELEMENTS) / 2)
    plots[f"{ELEMENTS[i]}_err"] = sns.kdeplot(
        data=refined_alloy_properties, 
        y=f"{ELEMENTS[i]}", 
        x="percent_error_e", 
        ax=axis[k,j], 
        fill=True, 
        cmap="cividis_r",
    )
    plots[f"{ELEMENTS[i]}_err"].set_xlim([0, 2.5])
    plots[f"{ELEMENTS[i]}_err"].set_ylim(bottom=0)

plots["fe_err"].set_ylim([60, 85])
plots["c_err"].set_ylim(top=0.5)
plots["mn_err"].set_ylim(top=1.2)
plots["si_err"].set_ylim(top=2.5)
plots["cr_err"].set_ylim(top=22.5)
plots["ni_err"].set_ylim(top=25)
plots["mo_err"].set_ylim(top=8)
plots["v_err"].set_ylim(top=2.25)
plots["n_err"].set_ylim(top=0.06)
plots["nb_err"].set_ylim(top=0.36)
plots["co_err"].set_ylim(top=22.5)
plots["w_err"].set_ylim(top=2.25)
plots["al_err"].set_ylim(top=1.4)
plots["ti_err"].set_ylim(top=2.5)

plt.tight_layout(w_pad=-3.5, h_pad=-1)
plt.show()

#
#The following is to validate the above model.
#

def cross_validation(n, k, data: pd.DataFrame):
    loop_range = n * k
    logged_std = np.zeros([3, loop_range])
    logged_r_squared = np.zeros([3, loop_range])
    logged_mape = np.zeros([3, loop_range])

    for i in range(0, loop_range, k - 1):
        samples: list[pd.DataFrame] = DataframeWriter.data_shuffler(data, k)
        control_sample: pd.DataFrame = samples[0].dropna()
        properties_array = np.array(control_sample["combined_properties"].values.tolist())

        for j in range(0, k - 1):
            A_learned = a_calc(samples[j+1])

            predicted_array = np.array(
                control_sample["combined_compositions"]
                .apply(lambda prop: np.matmul(A_learned, prop))
                .values
                .tolist()
            )

            r_squared =  r2_score(
                properties_array,
                predicted_array,
                multioutput="raw_values",
            )

            logged_r_squared[:, i + j] = r_squared

            logged_std[:, i + j] = calculate_std(properties_array, predicted_array)

            logged_mape[:, i + j] = calculate_mape(properties_array, predicted_array)

    errors_df = pd.DataFrame()
    errors_df["yield_stress_R^2"] = logged_r_squared[0]
    errors_df["tensile_stress_R^2"] = logged_r_squared[1]
    errors_df["elongation_R^2"] = logged_r_squared[2]
    errors_df["yield_stress_std"] = logged_std[0]
    errors_df["tensile_stress_std"] = logged_std[1]
    errors_df["elongation_std"] = logged_std[2]
    errors_df["yield_stress_mape"] = logged_mape[0]
    errors_df["tensile_stress_mape"] = logged_mape[1]
    errors_df["elongation_mape"] = logged_mape[2]

    return errors_df

errors_df = cross_validation(1000, 3, test_alloy_properties)

figure, axis = plt.subplots(3, 3)

sns.kdeplot(data=errors_df, x="yield_stress_R^2", clip=(0.0,1.0), ax=axis[0, 0])
sns.kdeplot(data=errors_df, x="tensile_stress_R^2", clip=(0.0,1.0), ax=axis[0, 1])
sns.kdeplot(data=errors_df, x="elongation_R^2", clip=(0.0,1.0), ax=axis[0, 2])
sns.kdeplot(data=errors_df, x="yield_stress_std", clip=(0.0, 500.0), ax=axis[1, 0])
sns.kdeplot(data=errors_df, x="tensile_stress_std", clip=(0.0, 500.0), ax=axis[1, 1])
sns.kdeplot(data=errors_df, x="elongation_std", clip=(0.0, 12.5), ax=axis[1, 2])
sns.kdeplot(data=errors_df, x="yield_stress_mape", clip=(0.0,100.0), ax=axis[2, 0])
sns.kdeplot(data=errors_df, x="tensile_stress_mape", clip=(0.0,100.0), ax=axis[2, 1])
sns.kdeplot(data=errors_df, x="elongation_mape", clip=(0.0,100.0), ax=axis[2, 2])

plt.show()

#
#The following is an algorithm to find an optimal minimum composition of Co and Ni for best mechanical properties
#and Ni content for best mechanical properties
#



