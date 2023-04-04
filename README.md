# Louvain_Priors
Louvain Algorithm with priors added to encourage certain community formations

# Description of important files
`data/pokemon_ground_truth.csv`
Contains features of all Pokemon (https://en.wikipedia.org/wiki/List_of_Pok%C3%A9mon), "each having unique designs, skills, and powers". No ground truths exist in terms of Pokemon community, so community clustering is an interesting concept over this data-set. Moreover, some partial ground-truths can be obtained because certain Pokemons are similar to others based on general design, attributes and other factors. We use these partial ground-truths as the validation data-set over our algorithm.

`priors.json`
Contains coupling and decoupling priors of indices referencing the node index in data-set. Coupling priors encourage two nodes to belong to the same community and decoupling priors discourage nodes from belonging to the final communities.

# To use
Simply run `python3 -m main` to compute communities over pokemon data-set (default) with priors given in `priors.json`
You should get the following result (baseline: random cluster assignment):
```
baseline purity score:  33.734939759036145
louvain own code with no-priors purity score :  55.42168674698795
louvain with priors purity score:  61.44578313253012
```
