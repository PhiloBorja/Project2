    # We import the libraries that we need
import pandas as pd
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    
    # Read the data from message and categories, and join in a dataframe(df)
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return messages.join(categories, on='id')
    #pass


def clean_data(df):
    
    # The disaster_messages is corrrectly separate in column. So we have to focus on the disaster_categories.
    # As we saw reading the data, the categories column in the disaster_categories are separate by ';'. So we have to split it.
    df_categories = pd.DataFrame(df['categories'].str.split(';', expand=True))
    
    # We have to select and extract the first row of the disaster_categories to create a list with the name of the column.
    # As we saw in the data, we found in the name of the column, for example 'offer-0'. So to clean it, we have to remove the last two position. In this case '-0'
    row = df_categories.iloc[0]
    col_categories = row.applymap(lambda x: x[:-2]).iloc[0,:]
    
    # Rename the columns of categories
    df_categories.columns = col_categories
 
    # As we have seen in the name of the column, we have to do the same with the values. But in this case we have to remove the first part and select just the last position.
    # In the case of 'offer-0', we select '0'. And We need to covert the value from string to numeric.
    for column in df_categories:
        df_categories[column] = df_categories[column].astype(str).str[-1]
    
        df_categories[column] = df_categories[column].astype(int)   
        
    # Now that we have clean the data, we remove the old column.
    df = df.drop(['categories'],axis=1)
    
    # Once removed, we dd the new one. 
    df = df.join(df_categories)
    
    # When we read the data we saw that disaster_categories has one column that has values that are not 0 or 1. The column is 'related' and the value is: 'related-2'. 
    # So we need to change 2 for 1.
    df['related'].replace(2, 1, inplace=True) 
    
    # Drop the rows with duplicates message
    df = df.drop_duplicates(subset='message')
    
    return df
    #pass 

def save_data(df, database_filename):
    
    # We write records stored in a DataFrame to a SQL database. In this case we sabe the values of the message column
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message', engine, index=False)  
    #pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()