import numpy as np
import pulp
import csv
import re
import argparse
import random

def read_csv(file):
    with open(file, 'r',encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)  # This will store the header row

        person_data = {}
        row_nombre = headers.index("Nombre completo")
        row_slots = headers.index("Seleccione aquellos horarios en los que estaría dispuesto a llevar la tutoría del curso.")
        
        for row in reader:
            name = row[row_nombre].strip()
            time_slots = row[row_slots].split(";")
            person_data[name] = time_slots

    return person_data
    

def main():

    parser = argparse.ArgumentParser(
        description='Asigne estudiantes a grupos de tutoría, para maximizar '\
                    'el alcance del programa.')
    
    parser.add_argument('-t','--tutores',default='tutores.csv',type=str,help='Archivo con posibilidades de tutores')
    parser.add_argument('-e','--estudiantes',default='estudiantes.csv',type=str,help='Archivo con posibilidades de estudiantes')
    parser.add_argument('-s','--seed', default=None, type=int,help='Semilla generadora de números aleatorios')
    parser.add_argument('-o','--output', default=None, type=str,help='Archivo CSV para dejar resultados')
    
    args = parser.parse_args()

    # Set the seed for the random number generator
    if args.seed is None:
        seed = random.randint(0, 1000000)
    else:
        seed = args.seed

    random.seed(seed)

    # ---------------------------------
    # Load data for tutors and students
    # ---------------------------------
    try:
        tutor_data = read_csv(file=args.tutores)
        student_data = read_csv(file=args.estudiantes)

    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise

    # Debug info: show the read data
    if False:
        print("Tutores y sus slots:")
        for i, (tutor, slots) in enumerate(tutor_data.items(),start=1):
            print(f"  {i}. {tutor}: {slots}\n")

        print("Estudiantes y sus slots:")
        for i,(student, slots) in enumerate(student_data.items(),start=1):
            print(f" {i}. {student}: {slots}")

    # Usually there are less data entries than lines in the CSV, because
    # some students answer twice or three times the form.
    print(f"\nHay {len(student_data)} estudiantes y {len(tutor_data)} tutores.")

    # -----------------------------------------
    # Convert the time slot strings to integers
    # -----------------------------------------
    
    # Step 1: Create a set of all available time slots
    all_slots = set()
    for data in [tutor_data, student_data]:
        for slots in data.values():
            all_slots.update(slots)

    # Step 2: Assign an integral index to every slot
    slot_to_index = {slot: index for index, slot in enumerate(all_slots)}

    # Step 3: Recreate the tutor_data and student_data using these indices
    def replace_slots_with_indices(data):
        return {name: [slot_to_index[slot] for slot in slots] for name, slots in data.items()}

    tutor_data = replace_slots_with_indices(tutor_data)
    student_data = replace_slots_with_indices(student_data)

    # Step 4: Create two dictionaries for the mappings
    index_to_slot = {index: slot for slot, index in slot_to_index.items()}


    # -------------------------
    # Linear programming 
    # -------------------------
        
    # Create the problem object
    prob = pulp.LpProblem("Tutor_Assignment", pulp.LpMaximize)

    # Create a binary variable for each student-tutor-slot combination
    x = pulp.LpVariable.dicts("x", (tutor_data.keys(), student_data.keys(), range(len(all_slots))), cat='Binary')

    # Create a binary variable for each tutor-time slot combination
    y = pulp.LpVariable.dicts("y", (tutor_data.keys(), range(len(all_slots))), cat='Binary')
    
    # Set the objective function
    total_students = pulp.lpSum(x[t][s][i] for t in tutor_data for s in student_data for i in range(len(all_slots)))
    prob += total_students 


    # Each tutor is assigned to exactly one time slot
    for t in tutor_data:
        prob += pulp.lpSum(y[t][i] for i in tutor_data[t]) == 1

    # Link x and y variables
    for t in tutor_data:
        for s in student_data:
            for i in tutor_data[t]:
                prob += x[t][s][i] <= y[t][i]
            
    # Each student is assigned to exactly one tutor at one time slot
    for s in student_data:
        prob += pulp.lpSum(x[t][s][i] for t in tutor_data for i in student_data[s]) <= 1
     

    # Students and tutors can only be assigned to available time slots
    for t in tutor_data:
        for s in student_data:
            for i in range(len(all_slots)):
                if i not in tutor_data[t] or i not in student_data[s]:
                    prob += x[t][s][i] == 0

                
    # Solve the problem
    # solver = pulp.CbcSolver()  # Choose a suitable solver
    prob.solve()

    # Print the optimal assignment
    for t in tutor_data:
        for s in student_data:
            for i in range(len(all_slots)):
                if pulp.value(x[t][s][i]) == 1:
                    print(f"Student {s} is assigned to tutor {t} at slot {index_to_slot[i]}")        

    print("\n")
                    
    # Print the students who were not assigned
    for s in student_data:
        if sum(pulp.value(x[t][s][i]) for t in tutor_data for i in range(len(all_slots))) == 0:
            print(f"Student {s} was not assigned to any group.")

    print("\n")
            
    # Statistics
            
    # Create a dictionary to store the number of students assigned to each tutor
    students_per_tutor = {t: 0 for t in tutor_data}

    # Count the number of students assigned to each tutor
    for t in tutor_data:
        for s in student_data:
            for i in tutor_data[t]:
                if pulp.value(x[t][s][i]) == 1:
                    students_per_tutor[t] += 1

    # Count the number of unassigned students
    unassigned_students = sum(1 for s in student_data if sum(pulp.value(x[t][s][i]) for t in tutor_data for i in student_data[s]) == 0)

    # Print the statistics
    for t in tutor_data:
        slot = next(index_to_slot[i] for i in tutor_data[t] if pulp.value(y[t][i]) == 1)
        print(f"Tutor {t} is assigned to slot {slot} with {students_per_tutor[t]} students.")

    print(f"There are {unassigned_students} unassigned students.")

    # Write a CSV with the results if so desired:

    if args.output is not None:
        print(f"Saving results in '{args.output}'")
        
        # Create a list of all tutors
        tutors = list(tutor_data.keys())

        # Create a dictionary to store the assignment of each student
        assignment = {s: {t: '' for t in tutors} for s in student_data}

        # Fill in the assignment dictionary with the results
        for t in tutor_data:
            for s in student_data:
                for i in tutor_data[t]:
                    if pulp.value(x[t][s][i]) == 1:
                        assignment[s][t] = index_to_slot[i]

        # Split names into first names and family names
        def split_name(name):
            parts = name.split()
            return ' '.join(parts[-2:]), ' '.join(parts[:-2])  # family names, first names

        # Create a new dictionary with split names and sorted by family names
        sorted_assignment = {split_name(s): assignment[s] for s in sorted(student_data, key=split_name)}

        # Write the results to a CSV file
        with open(args.output, 'w', newline='\n', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write the header
            writer.writerow(['Apellidos', 'Nombre'] + tutors)
            # Write the data
            for names, slots in sorted_assignment.items():
                writer.writerow(list(names) + list(slots.values()))    
                    
if __name__ == '__main__':
    main()
