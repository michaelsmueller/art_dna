import pandas as pd
from pathlib import Path

artists_df = pd.read_csv("raw_data/artists.csv")

# Normalize artist names (lowercase, underscores for matching)
artists_df["normalized_name"] = artists_df["name"].str.lower().str.replace(" ", "_")

# Normalize and split genres
artists_df["genre"] = artists_df["genre"].fillna("").apply(
    lambda g: [s.strip() for s in g.split(",") if s.strip()]
)

# Map normalized artist names to genres list
artist_to_genres = dict(zip(artists_df["normalized_name"], artists_df["genre"]))

# Load image filenames
image_dir = Path("raw_data/resized")
image_files = list(image_dir.glob("*.jpg"))

# Build final dataset
rows = []
for path in image_files:
    filename = path.name
    # Normalize filename artist name
    artist_name = "_".join(filename.split("_")[:-1]).lower()
    genres = artist_to_genres.get(artist_name, [])
    for genre in genres:
        rows.append({"filename": filename, "genre": genre})

# Save result
df = pd.DataFrame(rows)
df.to_csv("raw_data/labeled_data.csv", index=False)

# Summary
print(f"Saved {len(df)} rows")
print(f"Unique genres: {df['genre'].nunique()}")
print(df["genre"].value_counts())
