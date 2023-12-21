
import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
from streamlit_option_menu import option_menu

# The App
st.title('💊 MtbCA-Selec-Pred app')
st.info('MtbCA-Selec-Pred allows users to predict bioactivity and selectivity of a query molecule separately against the Mycobacterium tuberculosis carbonic anhydrase target protein isoforms, MtbCA1 and MtbCA2')



# loading the saved models
bioactivity_first_model = pickle.load(open('mtbca1_substructure.pkl', 'rb'))
bioactivity_second_model = pickle.load(open('mtbca1_2D.pkl', 'rb'))
bioactivity_third_model = pickle.load(open('mtbca2_substructure.pkl', 'rb'))
bioactivity_fourth_model = pickle.load(open('mtbca2_2D.pkl', 'rb'))

# Define the tabs
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs(['Main', 'About', 'What is Mycobacterium tuberculosis carbonic anhydrase (MtbCA)?', 'Dataset', 'Model performance', 'Python libraries', 'Citing us', 'Application Developers'])

with tab1:
    st.title('Application Description')
    st.success(
        " This module of [**MtbCA-Selec-Pred**](https://github.com/RatulChemoinformatics/MtbCA-Selec-Pred) has been built to predict bioactivity and identify potent inhibitors against Mycobacterium tuberculosis carbonic anhydrases, MtbCA1 and MtbCA2 using robust machine learning algorithms."
    )

# Define a sidebar for navigation
with st.sidebar:
    selected = st.selectbox(
        'Choose a prediction model',
        [
            'MtbCA1 prediction model using substructure fingerprints',
            'MtbCA1 prediction model using 1D and 2D molecular descriptors',
            'MtbCA2 prediction model using substructure fingerprints',
            'MtbCA2 prediction model using 1D and 2D molecular descriptors',
        ],
    )

# MtbCA1 prediction model using substructure fingerprints
if selected == 'MtbCA1 prediction model using substructure fingerprints':
    # page title
    st.title('Predict bioactivity of molecules against MtbCA1 using substructure fingerprints')

    # Molecular descriptor calculator
    def desc_calc():
        # Performs the descriptor calculation
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/SubstructureFingerprinter.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
        return href

    # Model building
    def build_model(input_data):
        # Apply model to make predictions
        prediction = bioactivity_first_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pKi')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    # Sidebar
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
        st.sidebar.markdown("""
        [Example input file](https://github.com/RatulChemoinformatics/QSAR/blob/main/predict.txt)
        """)

    if st.sidebar.button('Predict'):
        if uploaded_file is not None:
            load_data = pd.read_table(uploaded_file, sep=' ', header=None)
            load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

            st.header('**Original input data**')
            st.write(load_data)

            with st.spinner("Calculating descriptors..."):
                desc_calc()

            # Read in calculated descriptors and display the dataframe
            st.header('**Calculated molecular descriptors**')
            desc = pd.read_csv('descriptors_output.csv')
            st.write(desc)
            st.write(desc.shape)

            # Read descriptor list used in previously built model
            st.header('**Subset of descriptors from previously built models**')
            Xlist = list(pd.read_csv('mtbca1_substructure.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            st.write(desc_subset.shape)

            # Apply trained model to make prediction on query compounds
            build_model(desc_subset)
        else:
            st.warning('Please upload an input file.')

# MtbCA1 prediction model using 1D and 2D molecular descriptors
elif selected == 'MtbCA1 prediction model using 1D and 2D molecular descriptors':
    # page title
    st.title('Predict bioactivity of molecules against MtbCA1 using 1D and 2D molecular descriptors')

    # Molecular descriptor calculator
    def desc_calc():
        # Performs the descriptor calculation
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -2d -descriptortypes ./PaDEL-Descriptor/descriptors.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
        return href

    # Model building
    def build_model(input_data):
        # Apply model to make predictions
        prediction = bioactivity_second_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pKi')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    # Sidebar
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
        st.sidebar.markdown("""
        [Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
        """)

    if st.sidebar.button('Predict'):
        if uploaded_file is not None:
            load_data = pd.read_table(uploaded_file, sep=' ', header=None)
            load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

            st.header('**Original input data**')
            st.write(load_data)

            with st.spinner("Calculating descriptors..."):
                desc_calc()

            # Read in calculated descriptors and display the dataframe
            st.header('**Calculated molecular descriptors**')
            desc = pd.read_csv('descriptors_output.csv')
            st.write(desc)
            st.write(desc.shape)

            # Read descriptor list used in previously built model
            st.header('**Subset of descriptors from previously built models**')
            Xlist = list(pd.read_csv('mtbca1_2D.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            st.write(desc_subset.shape)

            # Apply trained model to make prediction on query compounds
            build_model(desc_subset)
        else:
            st.warning('Please upload an input file.')
            
            
# MtbCA2 prediction model using substructure fingerprints
if selected == 'MtbCA2 prediction model using substructure fingerprints':
    # page title
    st.title('Predict bioactivity of molecules against MtbCA2 using substructure fingerprints')

    # Molecular descriptor calculator
    def desc_calc():
        # Performs the descriptor calculation
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/SubstructureFingerprinter.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
        return href

    # Model building
    def build_model(input_data):
        # Apply model to make predictions
        prediction = bioactivity_third_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pKi')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    # Sidebar
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
        st.sidebar.markdown("""
        [Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
        """)

    if st.sidebar.button('Predict'):
        if uploaded_file is not None:
            load_data = pd.read_table(uploaded_file, sep=' ', header=None)
            load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

            st.header('**Original input data**')
            st.write(load_data)

            with st.spinner("Calculating descriptors..."):
                desc_calc()

            # Read in calculated descriptors and display the dataframe
            st.header('**Calculated molecular descriptors**')
            desc = pd.read_csv('descriptors_output.csv')
            st.write(desc)
            st.write(desc.shape)

            # Read descriptor list used in previously built model
            st.header('**Subset of descriptors from previously built models**')
            Xlist = list(pd.read_csv('mtbca2_substructure.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            st.write(desc_subset.shape)

            # Apply trained model to make prediction on query compounds
            build_model(desc_subset)
        else:
            st.warning('Please upload an input file.')
            
            
# MtbCA2 prediction model using 1D and 2D molecular descriptors
if selected == 'MtbCA2 prediction model using 1D and 2D molecular descriptors':
    # page title
    st.title('Predict bioactivity of molecules against MtbCA2 using 1D and 2D molecular descriptors')

    # Molecular descriptor calculator
    def desc_calc():
        # Performs the descriptor calculation
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -2d -descriptortypes ./PaDEL-Descriptor/descriptors.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
        return href

    # Model building
    def build_model(input_data):
        # Apply model to make predictions
        prediction = bioactivity_fourth_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pKi')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    # Sidebar
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
        st.sidebar.markdown("""
        [Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
        """)

    if st.sidebar.button('Predict'):
        if uploaded_file is not None:
            load_data = pd.read_table(uploaded_file, sep=' ', header=None)
            load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

            st.header('**Original input data**')
            st.write(load_data)

            with st.spinner("Calculating descriptors..."):
                desc_calc()

            # Read in calculated descriptors and display the dataframe
            st.header('**Calculated molecular descriptors**')
            desc = pd.read_csv('descriptors_output.csv')
            st.write(desc)
            st.write(desc.shape)

            # Read descriptor list used in previously built model
            st.header('**Subset of descriptors from previously built models**')
            Xlist = list(pd.read_csv('mtbca2_2D.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            st.write(desc_subset.shape)

            # Apply trained model to make prediction on query compounds
            build_model(desc_subset)
        else:
            st.warning('Please upload an input file.')
            
            
with tab2:
  coverimage = Image.open('Logo.png')
  st.image(coverimage)
with tab3:
  st.header('What are MtbCAs?')
  st.write('The genome of Mycobacterium tuberculosis (Mtb) encodes three β-carbonic anhydrases (CAs, EC 4.2.1.1) that are crucial for the life cycle of the bacterium. Studies conducted on mycobacterial β-CAs have shown that they are involved in the invasion and survival of pathogens in the host environment.')
with tab4:
  st.header('Dataset')
  st.write('''
    In our work, we retrieved a Mycobacterium tubaerculosis carbonic anhydrase (MtbCA1 and MtbCA2) dataset from the ChEMBL database. The data was curated and resulted in a non-redundant set of 124 and 42 MtbCA1 and MtbCA2 inhibitors respectively, which can be divided into:
    - 66 and 20 active compounds against MtbCA1 and MtbCA2 respectively
    - 17 and 16 inactive compounds against MtbCA1 and MtbCA2 respectively
    - 41 and 6 intermediate compounds against MtbCA1 and MtbCA2 respectively
    ''')
with tab5:
  st.header('Model performance')
  st.write('We selected two different molecular signatures namely substructure fingerprints and 1D 2D molecular descriptors for both MtbCA1 and MtbCA2 to build the web application. The correlation coefficient, RMSE, and MAE values for the substructure fingerprint model with respect to MtbCA1 inhibition was found to be 0.955, 0.3073, and 0.2367. The correlation coefficient, RMSE, and MAE values for the 1D and 2D molecular descriptor model with respect to MtbCA1 inhibition was found to be 0.9787, 0.2551, and 0.2042. The correlation coefficient, RMSE, and MAE values for the substructure fingerprint model with respect to MtbCA2 inhibition was found to be 0.9645, 0.2785, and 0.1948. The correlation coefficient, RMSE, and MAE values for the 1D and 2D molecular descriptor model with respect to MtbCA2 inhibition was found to be 0.9864, 0.1971, and 0.1553.')
with tab6:
  st.header('Python libraries')
  st.markdown('''
    This app is based on the following Python libraries:
    - `streamlit`
    - `pandas`
    - `rdkit`
    - `padelpy`
  ''')
with tab7:
  st.markdown('Aspatwar A, Kairys V, Rala S, Parikka M, Bozdag M, Carta F, Supuran CT, Parkkila S. Mycobacterium tuberculosis β-carbonic anhydrases: Novel targets for developing antituberculosis drugs. ***International Journal of Molecular Sciences***. DOI: https://doi.org/10.3390/ijms20205153.')
with tab8:
  st.markdown('Ratul Bhowmik, Ajay Manaithiya, Dr. Ashok Aspatwar. [***Infectious Disease Lab, Faculty of Medicine and Health Technology, Tampere University, Finland***] ')