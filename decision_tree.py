import numpy as np
import os
import graphviz
import math


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    partitions = {}
    i = 0
    for val in x:
        if val in partitions:
            partitions[val].append(i)
        else:
            partitions[val] = [i]
        i += 1
    return partitions
    raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    h = 0
    values, count = np.unique(y, return_counts=True)
    for c in count:
        h = h + (c/len(y)) * (math.log(c/len(y), 2))
    return -h
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    mutual_information = {}
    values, count = np.unique(x, return_counts=True)
    E_y = entropy(y)
    for v in values:
        new_y1 = y[np.where( x==v )]
        new_y2 = y[np.where( x!=v )]
        mutual_information[v] = E_y - ((len(new_y1)/len(y)) * entropy(new_y1) + (len(new_y2)/len(y)) * entropy(new_y2))
    return mutual_information
    raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    if (np.all(y == 1)): return 1
     
    elif (np.all(y == 0)): return 0
    
    elif ((len(attribute_value_pairs) == 0) or (depth == max_depth)):
        values, count = np.unique(y, return_counts=True)
        return int(values[np.where(count == max(count))][0])
    
    elif(depth == max_depth):
        values, counts = np.unique(y, return_counts = True) 
        return int(values[np.where(counts == max(counts))][0])
    
    else:
        information_gain = []
        old_attribute = []
        for i in (attribute_value_pairs):
            new_x = np.empty(x.shape[0])
            if(i[0] not in old_attribute):
                old_attribute.append(i[0])
                new_x = x[:,i[0]]
                information_gain.append(mutual_information(new_x,y))
                
        maxindex,val,maximum = 0,0,-10000
        for i in information_gain:
            for key,values in i.items():
                if( values>maximum ):
                    maxindex = information_gain.index(i)
                    val = key
                    maximum = values
        best_attribute = ((maxindex,val))
        
        x_with_best_attribute, y_with_best_attribute, x_without_best_attribute, y_without_best_attribute = [], [], [], []
    
        for i in range(len(x)):
            if(x[i][best_attribute[0]] == best_attribute[1]):
                x_with_best_attribute.append(x[i])
                y_with_best_attribute.append(y[i])
            else:
                x_without_best_attribute.append(x[i])
                y_without_best_attribute.append(y[i])
        
        del attribute_value_pairs[best_attribute[0]]
        tree_dictonary = {}
        tree_dictonary[(best_attribute[0],best_attribute[1],False)]=(id3(np.array(x_without_best_attribute), np.array(y_without_best_attribute), attribute_value_pairs, depth+1, max_depth))
        attribute_value_pairs.append(best_attribute) 
        tree_dictonary[(best_attribute[0],best_attribute[1],True)]=(id3(np.array(x_with_best_attribute), np.array(y_with_best_attribute), attribute_value_pairs, depth+1, max_depth))
        return tree_dictonary
    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    if (type(tree) == int): return tree
    for i in list(tree.keys()):
        if (x[i[0]] == i[1]):
            return predict_example(x, tree[list(tree.keys())[-1]])
        else:
            return predict_example(x, tree[i])
        
    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    
    error = (1/len(y_true)) * sum(y_true != y_pred)
    return error
    raise Exception('Function not yet implemented!')


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('monks_data/monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('monks_data/monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    # Intitialize (attribute, value) pairs to be an empty array and populate it
    attribute_value_pairs = []  
    for i in range(Xtrn.shape[1]):
        X = np.array(M[:,i+1])
        for j in range(len(np.unique(X,return_counts = True)[1])):
            attribute_value_pairs.append((i, np.unique(X)[j]))

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, attribute_value_pairs)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
  
  