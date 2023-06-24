def get_vim_editor_answer(max_score,best_individuals):
    profession_count = {}
    for individual in best_individuals:
        chromosome = individual.chromosome 
        for gene in chromosome:
            profession = gene['profession']
            editor = gene['editors']
            if editor == 'Vim':
                profession_count[profession] = profession_count.get(profession, 0) + 1
    return profession_count