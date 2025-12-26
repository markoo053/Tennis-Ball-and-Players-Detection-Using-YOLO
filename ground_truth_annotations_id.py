import pandas as pd

# 1. Učitajte vaš postojeći GT CSV fajl
file_path = 'ground_truth_annotations_adjusted.csv'
df = pd.read_csv(file_path)

# 2. Inicijalizacija i obrada loptice
df['player_id'] = 0  
df.loc[df['class_name'] == 'ball', 'player_id'] = 0

# 3. Filtriranje samo igrača
# Radimo direktno na DataFrame-u, a ne na kopiji, ali filtriramo po 'player'
player_mask = df['class_name'] == 'player'
players_df = df[player_mask].copy() # Pravimo kopiju da izbegnemo SettingWithCopyWarning

# 4. Funkcija za dodeljivanje ID-a unutar grupe (frejma)
def assign_player_ids_transform(group):
    # Sortirajte grupu po x1
    group_sorted = group.sort_values(by='x1', ascending=True)
    
    # Kreirajte seriju ID-eva (1, 2, 3...) iste dužine kao grupa
    # range(1, len(group_sorted) + 1) -> npr. [1, 2] za dva igrača
    ids = pd.Series(range(1, len(group_sorted) + 1), index=group_sorted.index)
    
    return ids

# 5. Primena transformacije za dodelu ID-eva
# Koristimo originalni index grupe, što je ključno za spajanje
# ID-evi se dodeljuju u novu kolonu privremenog DataFrame-a 'players_df'
players_df['player_id'] = players_df.groupby('frame_id', group_keys=False).apply(
    assign_player_ids_transform
)

# 6. Spajanje rezultata nazad u originalni DataFrame
# Updatujemo samo redove gde su bili igrači
df.loc[player_mask, 'player_id'] = players_df['player_id']

# 7. Postavite tip kolone 'player_id' kao integer
df['player_id'] = df['player_id'].astype(int)

# 8. Prikaz i čuvanje novog CSV fajla
print("\nPodaci sa dodeljenim player_id (Player 1 = manji x1, Player 2 = veći x1):")
print(df.head(10))

output_file_path = 'ground_truth_with_ids.csv'
df.to_csv(output_file_path, index=False)

print(f"\nNovi GT fajl sa ID-evima je sačuvan kao: {output_file_path}")