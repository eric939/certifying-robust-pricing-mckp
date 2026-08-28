# Dominick's scanner data pilot

Source: Kilts Center for Marketing, University of Chicago Booth School of
Business.

- Dataset page: <https://www.chicagobooth.edu/research/kilts/research-data/dominicks>
- Manual: `Dominicks Manual and Codebook - updated and revised` on the source
  page.
- Terms: academic research use; acknowledge the Kilts Center in every working
  paper or publication using the data.

The current pilot downloads only the cereal category:

| File | Role | SHA-256 |
|---|---|---|
| `cereals/wcer.zip` | Weekly store--UPC movement data | `2d4a59f88e4b97257c566f7c62fcccfdf9d5971891a617316ab21ea61c8d24c2` |
| `cereals/upccer.csv` | UPC dictionary downloaded and hashed for provenance; not read by the pilot computation | `affa589f080ab18a2edb2fa8f4cfb52b9212dec353bb49778e57f2b2403c5c6c` |

The movement archive expands to approximately 458 MB. Pipelines should stream
the CSV from the zip archive rather than creating an additional raw copy.

Important manual definitions:

- dollar sales are `price * move / qty`;
- `profit` is gross margin in percent, so an accounting unit cost can be
  reconstructed as unit price times `1 - profit/100`;
- this is average acquisition cost, not necessarily current replacement cost;
- records with `ok = 0` are suspect and must be excluded;
- promotion codes are incomplete, so missing promotion codes do not prove that
  a week was unpromoted.

These limitations must remain visible in the application analysis.
