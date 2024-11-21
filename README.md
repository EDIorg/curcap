# curcap

Historical Data Curation Trends and Capacity Planning

## Usage

1. Get the data
   2. Download the data curation log from EDI's Google Drive
   3. Request a CSV file containing data on the number of
        data packages published to the `edi` scope of EDI repository by the data curation
        team (i.e. principal users are those of each curation team member, and includes the `EDI` superuser. For example, see the file `curator_published_data_packages.csv` in this Git repository.
2. Run the `main.py`
   3. Install dependencies (see environment.yml and requirements.txt files)
   3. Configure the function calls at the bottom of `main.py`
   4. Run `main.py` to generate the plots