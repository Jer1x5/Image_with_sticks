import random
import numpy as np
from img_preprocessing import preprocess_image, bresenham_line
from visualize import visualize

POPULATION_SIZE = 200  # размер популяции
GENERATIONS = 1000  # количество поколений
IMAGE = preprocess_image("circle.jpeg") # ndarray (256, 256) состоит из нулей и 255, где 0 - черный пиксель, 255 - белый

# посчитать значение подходимости (fitness) для конкретного отрезка (особи)
def get_fitness(individ):
    current_value = 0
    for pixel in individ.pixels:
        if IMAGE[pixel[0], pixel[1]] == 0:
            current_value += 1
        else:
            current_value -= 0.1

    return current_value # считаете количество пересечений ваших палок и черных пикселей на данной картинке

class Individ:
    def __init__(self, l, angle, x, y):
        # гены, которые нужно менять
        self.l = l
        self.angle = angle  # degrees
        self.middle_x = x
        self.middle_y = y

        # свойства, которые нельзя менять
        angle_rad = np.radians(angle)
        x0 = int(self.middle_x + np.cos(angle_rad) * l/2)
        y0 = int(self.middle_y + np.sin(angle_rad) * l/2)
        x1 = int(self.middle_x - np.cos(angle_rad) * l/2)
        y1 = int(self.middle_y - np.sin(angle_rad) * l/2)
        self.pixels = bresenham_line(x0, y0, x1, y1)
        self.fitness = get_fitness(self)

def create_individ():
    # Здесь код для создания одного индивида в популяции
    # каждый индивид - [длина палки, наклон палки, x_середины, y_середины]
    l = random.randint(1, 5)  # длина палки от 5 до 40 пикселей
    angle = random.randint(0, 359)  # наклон в градусах
    x = random.randint(10, 216)
    y = random.randint(10, 216)
    individ = Individ(l, angle, x, y)

    return individ


# создаем всю популяцию
def create_population(size):  # size = POPULATION_SIZE
    population = []
    for item in range(size):
        population.append(create_individ())

    return population.copy()

def selection(population):
    # выбор 50% самых лучших индивидов из популяции
    sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
    return sorted_population[:POPULATION_SIZE//2]

def crossover(individ1, individ2):
    # скрещиваем гены первого и второго индивида
    # индивидов выбираем из лучших
    l = individ1.l
    angle = individ1.angle
    middle_x = individ1.middle_x
    middle_y = individ1.middle_y
    probability = random.random()
    if probability < 0.1:
        l = individ2.l
    elif probability < 0.2:
        l = (individ1.l + individ2.l)/2
    elif probability < 0.3:
        angle = individ2.angle
    elif probability < 0.4:
        angle = ((individ1.angle + individ2.angle)%360)/2
    elif probability < 0.5:
        middle_x = individ2.middle_x
    elif probability < 0.6:
        middle_x = (individ2.middle_x + individ1.middle_x)/2
    elif probability < 0.7:
        middle_y = individ2.middle_y
    elif probability < 0.8:
        middle_y = (individ2.middle_y + individ1.middle_y)/2
    elif probability < 0.9:
        middle_x = individ2.middle_y
        middle_y = individ2.middle_x
    else:
        middle_x = individ1.middle_y
        middle_y = individ1.middle_x

    new_individ = Individ(l, angle, middle_x, middle_y)
    return new_individ


def mutation(individ):
    # мутация отдельных генов индивида
    l = individ.l
    angle = individ.angle
    middle_x = individ.middle_x
    middle_y = individ.middle_y
    probability = random.random()
    if probability < 0.25:
        l = random.randint(5, 40)
    elif probability < 0.5:
        angle = random.randint(0, 359)
    elif probability < 0.75:
        middle_x = random.randint(40, 216)
    else:
        middle_y = random.randint(40, 216)

    new_individ = Individ(l, angle, middle_x, middle_y)
    return new_individ

def main():
    population = create_population(POPULATION_SIZE)
    for generation in range(GENERATIONS):
        best_individs = selection(population) # селекция
        crossovered = [crossover(random.choice(best_individs),random.choice(best_individs)) for _ in range(POPULATION_SIZE//4)]# тут несколько строк кода,
        # чтобы сделать скрещивание между лучшими особями
        # и создать 25% популяции скрещиванием
        mutated = [mutation(random.choice(best_individs)) for _ in range(POPULATION_SIZE//4)] # тут несколько строк кода,
        # чтобы сделать мутации лучшим
        # и создать 25% популяции мутацией
        population = best_individs + crossovered + mutated # размер популяции всегда фиксированный

        population = sorted(population, key=lambda x: x.fitness, reverse=True)
        print(f'generation {generation}, fitness: {sum([x.fitness for x in population])}')
    visualize(IMAGE.shape, population)

if __name__ == '__main__':
    main()