import numpy as np
import random
import copy
import uuid
from time import sleep
from tqdm import tqdm


class VimGeneticAlgorithm():
    def __init__(self,population_size = 100,number_of_parents = 20,crossover_method = "simple",thr_explore = 0.1,thr_greedy = 0.9,gene_mutation_percentage = 0.2,attribute_mutation_percentage = 0.3,attribute_mutation_quantity = 5):
        self.chromosome_length = 5
        self.attributes = {"colour":["Red","Yellow","Blue","White","Green"],
                          "profession":['Mathematician', 'Hacker', 'Analyst', 'Developer', 'Engineer'],
                          "languages" : ['Python', 'Java', 'JavaScript', 'C#', 'Other'],
                          "NoSQL" : ['Redis', 'MongoDB', 'HBase', 'Cassandra', 'Neo4J'],
                          "editors" : ['Vim', 'Brackets', 'Atom', 'Notepad++', 'Sublime_Text']
                          }        
        self.population_size = population_size        
        self.population = self.create_initial_population()           
        self.number_of_parents = number_of_parents
        self.gene_mutation_percentage = gene_mutation_percentage
        self.attribute_mutation_percentage = attribute_mutation_percentage
        self.attribute_mutation_quantity = attribute_mutation_quantity
        if self.number_of_parents > self.population_size:
            print(f"number of parents cant be greater than population. Using population_size/2 as number of parents: {self.population_size/2}")
            self.number_of_parents = self.population_size/2            
        if (self.number_of_parents % 2) != 0:
            print(f"Number of parents is odd. Using number of parents {self.number_of_parents + 1}")
            self.number_of_parents =  self.number_of_parents + 1
        else:        
            self.number_of_parents =  self.number_of_parents    
            
        self.number_of_parents = int(self.number_of_parents)
        
        self.crossover_method = crossover_method
        
        self.thr_explore = thr_explore
        self.thr_greedy = thr_greedy
        self.explore_generation = None
        self.greedy_generation = None
        
    def create_initial_population(self):
        population = []
        for n in range(self.population_size):
            chromosome = self.fabricate_chromosome()
            population.append(Individual(chromosome))
        return population
    
    def aptitude_function(self,individual):        
        score = 0
        
        #1 The Mathematician lives in the red house
        for gene in individual.chromosome:    
            score += gene["profession"] == "Mathematician" and gene["colour"] == "Red"    
        
        #2 The hacker programs in Python
        for gene in individual.chromosome:    
            score += (gene["profession"] == "Hacker" and gene["languages"] == "Python")
            
        #3 The Brackets is used in the green house
        for gene in individual.chromosome:    
            score += (gene["editors"] == "Brackets" and gene["colour"] == "Green")
            
        #4 The analyst uses Atom
        for gene in individual.chromosome:    
            score += (gene["profession"] == "Analyst" and gene["editors"] == "Atom")
        
        #5 The green house is to the right of the white house
        colours = [i["colour"] for i in individual.chromosome]  
        for colour_index in range(len(colours)):
            if colours[colour_index] == "White":
                try:
                    score += colours[colour_index + 1] == "Green"  
                    
                except IndexError:
                    pass
               
        #6 The person using Redis programs in Java        
        for gene in individual.chromosome:    
            score += (gene["NoSQL"] == "Redis" and gene["languages"] == "Java")
        
        #7 Cassandra is used in the yellow house
        for gene in individual.chromosome:    
            score += (gene["NoSQL"] == "Cassandra" and gene["colour"] == "Yellow")  
            
        #8 Notepad++ is used in the middle house        
        score += individual.chromosome[2]["editors"] == "Notepad++"   #no hardcodear el medio
        
        #9 The Developer lives in the first house
        score += individual.chromosome[0]["profession"] == "Developer"
       
        #10 The person who uses HBase lives next door to the person who programs JavaScript   
        for database_index in range(len(individual.chromosome)):
            if individual.chromosome[database_index]["NoSQL"] == "HBase":
                try:
                    score += individual.chromosome[database_index + 1]["languages"] == "JavaScript"                    
                    if database_index - 1 >= 0:
                        score += individual.chromosome[database_index - 1]["languages"] == "JavaScript"
                except IndexError:
                    pass        
       
        #11 The person using Cassandra is a neighbor of the person programming in C#   
        for database_index2 in range(len(individual.chromosome)):
            if individual.chromosome[database_index2]["NoSQL"] == "Cassandra":
                try:
                    score += individual.chromosome[database_index2 + 1]["languages"] == "C#"
                    if database_index2 -1 >= 0:
                        score += individual.chromosome[database_index2 - 1]["languages"] == "C#"
                except IndexError:
                    pass    
                        
        #12  The person using Neo4J uses Sublime Text
        for gene in individual.chromosome:    
            score += (gene["NoSQL"] == "Neo4J" and gene["editors"] == "Sublime_Text")       
        
        #13 The Engineer uses MongoDB
        for gene in individual.chromosome:    
            score += (gene["profession"] == "Engineer" and gene["NoSQL"] == "MongoDB")
     
        
        #14 The developer lives in the blue house
        for gene in individual.chromosome:    
            score += (gene["profession"] == "Developer" and gene["colour"] == "Blue")    
        
        #15 Repeated attributes set score to 0
        colour = []
        profession = []
        languages = []
        NoSQL = []
        editors = []

        for gene in individual.chromosome:
            colour.append(gene["colour"])
            profession.append(gene["profession"])
            languages.append(gene["languages"])
            NoSQL.append(gene["NoSQL"])
            editors.append(gene["editors"])
        all_attributes = [colour,profession,languages,NoSQL,editors]        
        
        if any([len(lst) != len(set(lst)) for lst in all_attributes]):           
            score = 0            
        return score
    
    def crossover(self,parents):
        parent_x = parents[0]
        parent_y = parents[1]
        
        if self.crossover_method == "simple":
            cut_point = random.randint(0,self.chromosome_length)
            chromosome_child_a = parent_x.chromosome[:cut_point] + parent_y.chromosome[cut_point:]
            chromosome_child_b = parent_x.chromosome[cut_point:] + parent_y.chromosome[:cut_point]
          
        if self.crossover_method == "Multi_Point_Crossover":        
            cut_points = sorted(random.sample(range(self.chromosome_length), 2))
            chromosome_child_a = parent_x.chromosome[:cut_points[0]] +  parent_y.chromosome[cut_points[0]:cut_points[1]]  +  parent_x.chromosome[cut_points[1]:]
            chromosome_child_b = parent_y.chromosome[:cut_points[0]] +  parent_x.chromosome[cut_points[0]:cut_points[1]]  +  parent_y.chromosome[cut_points[1]:]  
        
        if self.crossover_method == "Binomial_Mask_Crossover":
            boolean_mask = np.random.choice([True, False], size=self.chromosome_length, p=[0.5, 0.5])
            
            chromosome_child_a = []
            chromosome_child_b = []

            for gene_index in range(self.chromosome_length):
                if boolean_mask[gene_index] == True:
                    chromosome_child_a.append(parent_y.chromosome[gene_index])
                    chromosome_child_b.append(parent_x.chromosome[gene_index])
                else:
                    chromosome_child_a.append(parent_x.chromosome[gene_index])
                    chromosome_child_b.append(parent_y.chromosome[gene_index])
                    
        child_a = Individual(chromosome_child_a)
        child_b = Individual(chromosome_child_b)         
        
        return [child_a,child_b]            
    
    def fabricate_chromosome(self):
        copied_attributes = copy.deepcopy(self.attributes)
        chromosome = []
        for gene_index in range(self.chromosome_length):
            gene = {}            
            for attribute in copied_attributes:
                gene[attribute] = copied_attributes[attribute].pop(random.randrange(len(copied_attributes[attribute])))                                     
            chromosome.append(gene)        
        return chromosome     
         
    def score_population(self,population):
        scored_population = {}
        for individual in population:            
            scored_population[individual] = self.aptitude_function(individual)
        return scored_population
    
    def select(self,scored_population,current_generation):    
        selected_individuals = []    

        if current_generation <= self.explore_generation:       
            for parent_index in range(self.number_of_parents):
                selected_individuals.append(random.choice(list(scored_population.keys()))) 

        elif current_generation > self.explore_generation and current_generation < self.greedy_generation:
            values = np.array(list(scored_population.values()))
            max_value = np.max(values)
            probabilities = np.exp(values - max_value) / np.sum(np.exp(values - max_value))
            for parent_index in range(self.number_of_parents):
                sampled_object = np.random.choice(list(scored_population.keys()), p=probabilities, replace=False)
                selected_individuals.append(sampled_object)  

        elif current_generation >= self.greedy_generation:    
            sorted_population = sorted(scored_population.items(), key=lambda x:x[1],reverse=True)        
            for parent_index in range(self.number_of_parents):
                selected_individuals.append(sorted_population[parent_index][0]) 

        return  selected_individuals 
    
    
    def mutate_individual_attributes(self,individual):        
        for mutation in range(self.attribute_mutation_quantity):
            attributes = list(self.attributes.keys())
            mutation_attribute = random.choice(attributes)
            mutation_gene1 = random.randint(0,self.chromosome_length - 1)
            mutation_gene2 = random.randint(0,self.chromosome_length - 1)
            while mutation_gene1 == mutation_gene2:
                mutation_gene2 = random.randint(0,self.chromosome_length - 1)    

            attribute_1 = individual.chromosome[mutation_gene1][mutation_attribute]
            attribute_2 = individual.chromosome[mutation_gene2][mutation_attribute]

            individual.chromosome[mutation_gene1][mutation_attribute] = attribute_2
            individual.chromosome[mutation_gene2][mutation_attribute] = attribute_1   
            
            
    def mutate_individual_gene(self,individual):        
        mutation_gene1 = random.randint(0,self.chromosome_length - 1)
        mutation_gene2 = random.randint(0,self.chromosome_length - 1)
        aux = individual.chromosome[mutation_gene1]
        individual.chromosome[mutation_gene1] = individual.chromosome[mutation_gene2]
        individual.chromosome[mutation_gene2] = aux         
    
    def mutate(self, population):
        mutated_population = []

        gene_individuals_to_mutate = int(len(population) * self.gene_mutation_percentage)
        attribute_individuals_to_mutate = int(len(population) * self.attribute_mutation_percentage)

      
        for individual in population[:gene_individuals_to_mutate]:
            mutated_individual = copy.deepcopy(individual)
            self.mutate_individual_gene(mutated_individual)
            mutated_population.append(mutated_individual)

      
        for individual in population[gene_individuals_to_mutate:(gene_individuals_to_mutate + attribute_individuals_to_mutate)]:
            mutated_individual = copy.deepcopy(individual)
            self.mutate_individual_attributes(mutated_individual)
            mutated_population.append(mutated_individual)
       
        mutated_population += population[(gene_individuals_to_mutate + attribute_individuals_to_mutate):]

        return mutated_population

    
    def reproduce(self,selected_individuals): 
        children = []
        parents_pairs = [selected_individuals[i:i + 2] for i in range(0, len(selected_individuals), 2)]
        for pair in parents_pairs:
            children += self.crossover(pair)
        return children
        
    def generation_step(self, current_generation):
        scored_population = self.score_population(self.population)
        parents = self.select(scored_population, current_generation)
        children = self.reproduce(parents)
        sorted_population = sorted(scored_population.items(), key=lambda x: x[1])      
        filtered_population = [i[0] for i in sorted_population[len(children):]]
        self.population = self.mutate(filtered_population + children)
        
    def fit(self,generations, cut_thr = 0.5):
        self.explore_generation = round(self.thr_explore * generations)
        self.greedy_generation = round(self.thr_greedy * generations)    
        amount_cut_individuals = self.population_size * cut_thr
        for i in tqdm(range(generations)):
            self.generation_step(current_generation = i)   
            generation_scored_population = self.score_population(self.population)
            generation_sorted_population = sorted(generation_scored_population.items(), key=lambda x: x[1],reverse=True)           
            best_individuals = [individual for individual, score in generation_sorted_population if score == 14]   
            if len(best_individuals) >= amount_cut_individuals:
                print("Reached cut condition")    
                print(f"Amount of individuals with max score: {len(best_individuals)}")        
                break
                
    def get_best_score_and_individuals(self):      
        scored_population = self.score_population(self.population)
        sorted_population = sorted(scored_population.items(), key=lambda x: x[1],reverse=True) 
        max_score = max([score for _, score in sorted_population])
        best_individuals = [individual for individual, score in sorted_population if score == max_score]   
        return max_score,best_individuals                   
                
class Individual():    
    def __init__(self,chromosome):   
        self.chromosome = chromosome