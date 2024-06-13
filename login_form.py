import streamlit as st
import pandas as pd


users_path = 'users.xlsx'


def login_form():
    
    st.title('Sign in')
    
    name = st.text_input('Name')
    password = st.text_input('Password', type='password')
    
    login_btn = st.button('Login')
    
    if login_btn:
        if check_name_and_password(name, password):
            return name
        else:
            st.warning('Incorrect username or password')
        
    
def check_name_and_password(name, password):
    users_df = pd.read_excel(users_path)
    
    for i, row in users_df.iterrows():
        if row['username'] == name and row['password'] == password:
            return True
    
    return False



def check_user_name(name):
    users_df = pd.read_excel(users_path)
    
    if name not in users_df['username'].values:
        return True
    
    return False