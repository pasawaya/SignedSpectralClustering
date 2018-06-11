import os
import re
import ssc
import nltk
import matplotlib.patches as mpatches

import json
import math
import random as rand
import mord
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer

from sklearn import linear_model, datasets, svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, minmax_scale, scale

from scipy.spatial.distance import cosine
from scipy.sparse import dok_matrix, vstack, csr_matrix, save_npz, load_npz


class AffinityMatrix(object):
    def __init__(self, root, model, threshold=0.8, should_lemmatize=True):
        self.model = model
        self.threshold = threshold
        self.should_lemmatize = should_lemmatize

        print("Loading recipe data...")
        self.recipes, self.ratings = self.load_recipe_data(root)
        self.rating_threshold = np.mean(self.ratings)

        print("Generating affinity matrix...")
        self.w, self.labels, self.indices = self.create_affinity_matrix()

    def tokenize(self, contents):
        tokens = re.compile("\\s+|[,;.'-]\s*").split(contents.lower())
        if self.should_lemmatize:
            net = WordNetLemmatizer()
            tokens = [net.lemmatize(x) for x in tokens]
        return tokens

    def load_recipe_data(self, root):
        directory = os.fsencode(root)

        titles = set()
        recipes = []
        ratings = []
        frequencies = {}
        net = WordNetLemmatizer()

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".json"):
                path = root + '/' + filename
                with open(path, 'r') as handle:
                    contents = handle.read().replace('}', '},').replace('\n', '')
                    contents = '{"recipes": [' + contents[:-1] + ']}'
                    obj = json.loads(contents)

                for recipe in obj['recipes']:
                    title = recipe['title']
                    rating = float(recipe['rating'])
                    if title not in titles and rating > 0.0:
                        titles.add(title)
                        ratings.append(rating)
                        ingredients = []
                        for ingredient in recipe['ingredients']:
                            tokens = nltk.word_tokenize(ingredient)
                            tagged = nltk.pos_tag(tokens)
                            components = [x[0].lower() for x in tagged]
                            tags = [x[1] for x in tagged]

                            nouns = []
                            for component, tag in zip(components, tags):
                                if tag == 'NN' or tag == 'NNS':
                                    nouns.append(net.lemmatize(component))

                            if nouns:
                                ingredient = " ".join(nouns)
                                ingredients.append(ingredient)
                                if ingredient not in frequencies:
                                    frequencies[ingredient] = 0
                                frequencies[ingredient] += 1
                        recipes.append(ingredients)

        fout = "ing.txt"
        fo = open(fout, "w")
        for k, v in frequencies.items():
            fo.write(str(k)+'\n')
        fo.close()

        sorted_ingredients = sorted(frequencies, key=frequencies.get)
        sorted_index = math.floor((len(sorted_ingredients) * self.threshold))
        trimmed_ingredients = set(sorted_ingredients[sorted_index:-1])
        cleaned_recipes = []
        cleaned_ratings = []

        for recipe, rating in zip(recipes, ratings):
            for ingredient in recipe:
                if ingredient not in trimmed_ingredients or '®' in ingredient:
                    recipe.remove(ingredient)

            if recipe:
                cleaned_recipes.append(recipe)
                cleaned_ratings.append(rating)
        return cleaned_recipes, cleaned_ratings

    def create_affinity_matrix(self):
        indices = {}
        labels = []
        index = 0

        for recipe in self.recipes:
            for ingredient in recipe:
                if ingredient not in indices:
                    indices[ingredient] = index
                    labels.append(ingredient)
                    index += 1

        n = len(indices.keys())
        m = len(self.recipes)

        if self.model == 'ratings':
            w = dok_matrix((n, n), dtype=float)
            counts = np.zeros((n, n))
            for recipe, rating in zip(self.recipes, self.ratings):
                for ing1 in recipe:
                    for ing2 in recipe:
                        idx1 = indices[ing1]
                        idx2 = indices[ing2]
                        w[idx1, idx2] += rating - self.rating_threshold
                        w[idx2, idx1] += rating - self.rating_threshold
                        counts[idx1, idx2] += 1
                        counts[idx2, idx1] += 1

            w = w.todense()
            w = np.where(counts > 0, w / np.sqrt(counts), 0)
            bound = max(abs(np.amax(w)), abs(np.amin(w)))
            w = w / bound

        elif self.model == 'npmi' or self.model == 'pmi':
            p_a = np.full((n, 1), 1e-14)
            p_ab = np.full((n, n), 1e-14)

            for recipe in self.recipes:
                for ing1 in recipe:
                    idx1 = indices[ing1]
                    p_a[idx1] += 1.0
                    for ing2 in recipe:
                        idx2 = indices[ing2]
                        p_ab[idx1, idx2] += 1.0
                        p_ab[idx2, idx1] += 1.0
            p_a /= m
            p_ab /= 2 * m

            if self.model == 'npmi':
                w = (np.log(p_ab / (p_a * p_a.T))) / (-1 * np.log(p_ab))
            else:
                w = np.log(p_ab / (p_a * p_a.T))
                minmax_scale(w, feature_range=(-1, 1), copy=False)

        else:
            w_rating = dok_matrix((n, n), dtype=float)
            counts = np.zeros((n, n))
            for recipe, rating in zip(self.recipes, self.ratings):
                for ing1 in recipe:
                    for ing2 in recipe:
                        idx1 = indices[ing1]
                        idx2 = indices[ing2]
                        w_rating[idx1, idx2] += rating - self.rating_threshold
                        w_rating[idx2, idx1] += rating - self.rating_threshold
                        counts[idx1, idx2] += 1
                        counts[idx2, idx1] += 1

            w_rating = w_rating.todense()
            w_rating = np.where(counts > 0, w_rating / np.sqrt(counts), 0)
            bound = max(abs(np.amax(w_rating)), abs(np.amin(w_rating)))
            w_rating = w_rating / bound

            p_a = np.full((n, 1), 1e-14)
            p_ab = np.full((n, n), 1e-14)

            for recipe in self.recipes:
                for ing1 in recipe:
                    idx1 = indices[ing1]
                    p_a[idx1] += 1.0
                    for ing2 in recipe:
                        idx2 = indices[ing2]
                        p_ab[idx1, idx2] += 1.0
                        p_ab[idx2, idx1] += 1.0
            p_a /= m
            p_ab /= 2 * m

            w_npmi = (np.log(p_ab / (p_a * p_a.T))) / (-1 * np.log(p_ab))
            w = (w_rating + w_npmi)/2

        np.fill_diagonal(w, 0)
        w = csr_matrix(w)
        return w, labels, indices

    def statistics(self):
        inverse_indices = {v: k for k, v in self.indices.items()}

        max_col = np.argmax(np.max(self.w, axis=1))
        max_row = np.argmax(np.max(self.w, axis=0))

        max_ing_1 = inverse_indices[max_col]
        max_ing_2 = inverse_indices[max_row]

        min_col = np.argmin(np.min(self.w, axis=1))
        min_row = np.argmin(np.min(self.w, axis=0))

        min_ing_1 = inverse_indices[min_col]
        min_ing_2 = inverse_indices[min_row]

        print("Statistics:")
        print("\t Size: " + str(self.w.shape))
        print("\t Max: " + str(self.w.max()) + " (" + max_ing_1 + ", " + max_ing_2 + ")")
        print("\t Mean: " + str(self.w.mean()))
        print("\t Min: " + str(self.w.min()) + " (" + min_ing_1 + ", " + min_ing_2 + ")")


def print_clusters(n_clusters, indices, labels):
    for cluster in range(0, n_clusters):
        print("\n")
        print("Group %d" % cluster)
        for i in range(0, len(indices)):
            if indices[i] == cluster:
                print(labels[i])


def show_heatmap(affinity, ingredients, axis_label):
    heatmap = np.zeros((len(ingredients), len(ingredients)))
    for i in range(len(ingredients)):
        for j in range(len(ingredients)):
            ing1 = ingredients[i]
            ing2 = ingredients[j]
            idx1 = affinity.indices[ing1]
            idx2 = affinity.indices[ing2]
            heatmap[i, j] = affinity.w[idx1, idx2]
            heatmap[j, i] = affinity.w[idx1, idx2]

    heatmap = np.round(heatmap, decimals=2)
    bound = max(abs(np.amax(heatmap)), abs(np.amin(heatmap)))

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(heatmap, cmap=plt.cm.seismic_r, vmin=-1, vmax=1)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(axis_label, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(ingredients)))
    ax.set_yticks(np.arange(len(ingredients)))
    ax.set_xticklabels(ingredients)
    ax.set_yticklabels(ingredients)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(len(ingredients)):
        for j in range(len(ingredients)):
            if heatmap[i, j] == 0:
                ax.text(j, i, heatmap[i, j], ha="center", va="center", color="xkcd:charcoal")
            else:
                ax.text(j, i, heatmap[i, j], ha="center", va="center", color="w")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    recipe_data = 'recipe_data'
    matrix = AffinityMatrix(recipe_data, model='npmi', threshold=0.0)
    matrix.statistics()
    n = len(matrix.labels)

    frequencies = np.zeros(n)

    for recipe in matrix.recipes:
        for ingredient in recipe:
            index = matrix.indices[ingredient]
            frequencies[index] += 1

    sorted_indices = np.argsort(frequencies)[::-1][:n]
    frequencies = frequencies[sorted_indices]
    labels = np.array(matrix.labels)[sorted_indices]

    plt.xlabel('Ingredient')
    plt.ylabel('Count')
    plt.plot(frequencies, color='#ff6666')
    plt.rcParams.update({'font.size': 8})

    n_labels = 5
    for i in np.arange(5, n - 5, step=int(n/4)):
        # rand_idx = rand.randint(0, frequencies.shape[0])
        plt.text(i, frequencies[i], labels[i])
    plt.show()

    # heatmap_ingredients = ['flour', 'sugar', 'milk', 'bacon', 'egg', 'vodka', 'tortilla', 'salsa', 'lime', 'tequila']
    # show_heatmap(matrix, heatmap_ingredients, axis_label='Rating-NPMI Average')

    cluster_sizes = []
    train_accuracies = []
    test_accuracies = []

    min_size = 1
    max_size = n
    iterations = 7
    incr = int((max_size-min_size)/iterations)
    # for k in np.arange(min_size, max_size, incr):
    for k in [534]:
        print("\nk = " + str(k))
        print("----------------------------")
        print('\tClustering ' + str(n) + ' ingredients into ' + str(k) + ' clusters...')
        cluster_sizes.append(k)

        file_name = str(k) + '.npy'
        path = "/Users/philippe/PycharmProjects/SSC/"
        if os.path.isfile(path + file_name):
            idx = np.load(file_name)
        else:
            [idx, XXc] = ssc.sncut(matrix.w, n_clusters=k, fast=True, method='hard')
            np.save(file_name, idx)

        clusters = dict(zip(matrix.labels, idx))
        print_clusters(k, idx, matrix.labels)

        print('\tExtracting features from recipe data...')
        m = len(matrix.recipes)
        recipe_idx = 0
        n_features = 3 * k
        X = np.zeros((m, n_features))
        Y = np.zeros(m)
        for recipe, rating in zip(matrix.recipes, matrix.ratings):
            feature = np.zeros(n_features)

            for ing1 in recipe:
                cluster1 = clusters[ing1]

                # |C|
                feature[cluster1] += 1

                for ing2 in recipe:
                    if ing1 != ing2:
                        cluster2 = clusters[ing2]
                        idx1 = matrix.indices[ing1]
                        idx2 = matrix.indices[ing2]

                        # assoc(C)
                        if cluster1 == cluster2:
                            feature[k + cluster1] += matrix.w[idx1, idx2]

                        # cut(C)
                        else:
                            feature[2 * k + cluster1] += matrix.w[idx1, idx2]
                            feature[2 * k + cluster2] += matrix.w[idx1, idx2]

            X[recipe_idx, :] = feature
            Y[recipe_idx] = int(round(rating))
            recipe_idx += 1

        # x_train = X
        # y_train = Y

        train_size = int(7 * m / 8)
        x_train = X[0:train_size, :]
        y_train = Y[0:train_size]

        x_test = X[train_size:m, :]
        y_test = Y[train_size:m]

        # Cs = [0.001, 0.01, 0.1, 1, 10]
        # gammas = [0.001, 0.01, 0.1, 1]
        # param_grid = {'C': Cs, 'gamma': gammas}
        # classes, class_counts = np.unique(y_train, return_counts=True)
        # clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight=dict(zip(classes, class_counts))), param_grid, cv=3)
        # clf.fit(x_train, y_train)
        # print(clf.best_params_)
        # print(clf.best_score_)

        print('\tTraining SVM...')
        classes, class_counts = np.unique(y_train, return_counts=True)
        clf_svm = svm.SVC(class_weight=dict(zip(classes, class_counts)), C=0.01)
        clf_svm.fit(x_train, y_train)
        pred_test = clf_svm.predict(x_test)
        svm_train_accuracy = accuracy_score(y_train, clf_svm.predict(x_train))
        svm_test_accuracy = accuracy_score(y_test, pred_test)
        print("\n\tSVM Train Accuracy: " + str(svm_train_accuracy))
        print("\tSVM Test Accuracy: " + str(svm_test_accuracy))
        train_accuracies.append(svm_train_accuracy)
        test_accuracies.append(svm_test_accuracy)
        print(confusion_matrix(y_test, pred_test, labels=np.unique(y_test)))

        # n_estimators = [int(x) for x in np.linspace(start=5, stop=200, num=10)]
        # max_depth = [int(x) for x in np.linspace(8, 64, num=8)]
        # random_grid = {'n_estimators': n_estimators, 'max_depth': max_depth}
        #
        # classes, class_counts = np.unique(y_train, return_counts=True)
        # forest = RandomForestClassifier(bootstrap=True, class_weight=dict(zip(classes, class_counts)),
        #                                 max_features='auto')
        # rf_random = RandomizedSearchCV(estimator=forest,
        #                                param_distributions=random_grid,
        #                                n_iter=20,
        #                                cv=3,
        #                                verbose=2,
        #                                n_jobs=-1)
        # rf_random.fit(x_train, y_train)
        # print(rf_random.best_params_)
        # print(rf_random.best_score_)

        # print('\tTraining random forest classifier...')
        # classes, class_counts = np.unique(y_train, return_counts=True)
        # forest = RandomForestClassifier(n_estimators=178,
        #                                 bootstrap=True,
        #                                 max_depth=40,
        #                                 max_features='auto',
        #                                 class_weight=dict(zip(classes, class_counts)))
        # forest = forest.fit(x_train, y_train)
        # pred_test = forest.predict(x_test)
        # forest_train_accuracy = accuracy_score(y_train, forest.predict(x_train))
        # forest_test_accuracy = accuracy_score(y_test, pred_test)
        # print("\n\tRandom Forest Train Accuracy: " + str(forest_train_accuracy))
        # print("\tRandom Forest Test Accuracy: " + str(forest_test_accuracy))
        # train_accuracies.append(forest_train_accuracy)
        # test_accuracies.append(forest_test_accuracy)

        # classes, class_counts = np.unique(y_train, return_counts=True)
        # param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        # clf = GridSearchCV(linear_model.LogisticRegression(penalty='l2', class_weight=dict(zip(classes,
        #                                                                                        class_counts))),
        #                    param_grid, cv=4)
        # clf.fit(x_train, y_train)
        # print(clf.best_params_)
        # print(clf.best_score_)

        # print('\tTraining logistic regression classifier...')
        # classes, class_counts = np.unique(y_train, return_counts=True)
        # reg = linear_model.LogisticRegression(C=0.001, class_weight=dict(zip(classes, class_counts)))
        # reg.fit(x_train, y_train)
        # pred_test = reg.predict(x_test)
        # log_train_accuracy = accuracy_score(y_train, reg.predict(x_train))
        # log_test_accuracy = accuracy_score(y_test, pred_test)
        # print("\n\tLogistic Regression Train Accuracy: " + str(log_train_accuracy))
        # print("\tLogistic Regression Test Accuracy: " + str(log_test_accuracy))
        # train_accuracies.append(log_train_accuracy)
        # test_accuracies.append(log_test_accuracy)
        # print(confusion_matrix(y_test, pred_test, labels=np.unique(y_test)))

    plt.plot(cluster_sizes, train_accuracies, 'r-',
             cluster_sizes, test_accuracies, 'r:')
    plt.xlabel('# Clusters')
    plt.ylabel('% Accuracy')
    plt.show()
