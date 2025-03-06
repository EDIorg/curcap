# curcap

Historical Data Curation Trends and Capacity Planning

## Usage

1. Get the data
   
   1. Download the data curation log from EDI's Google Drive
   2. Request a CSV file containing data on the number of
        data packages published to the `edi` scope of EDI repository by the data curation
        team (i.e. principal users are those of each curation team member, and includes the `EDI` superuser). As of 2025-03-06 this list includes: EDI, csmith, sgrossmanclarke1, mobrien, kzollovenecek, jlemaire, bmcafee, skskoglund, cgries, gmaurer. See the file `curator_published_data_packages.csv` in this Git repository for formatting.
      
2. Run `main.py`

   1. Install dependencies (see environment.yml and requirements.txt files)
   2. Configure the function calls at the bottom of `main.py`
   3. Run `main.py` to generate the plots
