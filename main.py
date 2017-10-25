import sys
import logging
import math

from scipy.io import arff
from scipy.stats import chi2

nodes = 0


def main():
    logging.info("---Starting application---")
    data, meta = arff.loadarff("./test_data/training_subsetD.arff")
    testing_data, testing_meta = arff.loadarff("./test_data/testingD.arff")

    logging.info("---Data was loaded successfully---")
    attr_names = meta._attrnames
    print("Class" in attr_names)
    attr_names.remove("Class")
    print("Class" in attr_names)

    confidence_levels = [0.99, 0.95, 0.8, 0.5, 0.0]
    #confidence_levels = [0.95]

    for confidence in confidence_levels:
        metrics = ID3TreeNodeMetrics()
        root = ID3TreeNode(data, 'Class', b'True', b'False', attr_names, meta._attributes, confidence, metrics)
        logging.info("---ID3 decision tree was created successfully---")

        print("True nodes: %s" % str(metrics.true_nodes))
        print("False nodes: %s" % str(metrics.false_nodes))
        print("Condition nodes: %s" % str(metrics.condition_nodes))

        total = 0
        correct = 0
        true_positives = 0
        true_prediction_total = 0
        true_actual_total = 0
        for row in data:
            prediction = root.classify(row, 4)
            actual = row['Class']
            print("Prediction: %s; Actual: %s" % (prediction, actual))
            total += 1
            correct += 1 if prediction == actual else 0
            true_positives += 1 if prediction == actual and prediction == b'True' else 0
            true_prediction_total += 1 if prediction == b'True' else 0
            true_actual_total += 1 if actual == b'True' else 0

        print("Confidence: %s" % str(confidence))
        print("Correct: %d" % correct)
        print("Total: %d" % total)
        print("Total true actual: %s" % str(true_actual_total))
        print("Total true predictions: %s" % str(true_prediction_total))
        print("Total true correct predictions: %s" % str(true_positives))
        print("Percent: %s" % str(correct / total))
        #print("Precision: %s" % str(true_positives / true_prediction_total))
        #print("Recall: %s" % str(true_positives / true_actual_total))

        total = 0
        correct = 0
        true_positives = 0
        true_prediction_total = 0
        true_actual_total = 0
        for row in testing_data:
            prediction = root.classify(row)
            actual = row['Class']
            total += 1
            correct += 1 if prediction == actual else 0
            true_positives += 1 if prediction == actual and prediction == b'True' else 0
            true_prediction_total += 1 if prediction == b'True' else 0
            true_actual_total += 1 if actual == b'True' else 0
        print("Test Data Confidence: %s" % str(confidence))
        print("Test Data Correct: %d" % correct)
        print("Test Data Total: %d" % total)
        print("Total true actual: %s" % str(true_actual_total))
        print("Total true predictions: %s" % str(true_prediction_total))
        print("Total true correct predictions: %s" % str(true_positives))
        print("Test Data Percent: %s" % str(correct / total))
        #print("Test Data Precision: %s" % str(true_positives / true_prediction_total))
        #print("Test Data Recall: %s" % str(true_positives / true_actual_total))


class ID3TreeNodeMetrics:
    def __init__(self):
        self.true_nodes = 0
        self.false_nodes = 0
        self.condition_nodes = 0


class ID3TreeNode:
    true_nodes = 0
    false_nodes = 0
    condition_nodes = 0

    def __init__(self, data, target_attribute, target_value, negative_value, attributes, attributes_map, confidence, metrics, forced_label = None):
        print("CREATING NODE: %s" % ID3TreeNode.condition_nodes)
        ID3TreeNode.condition_nodes += 1

        # Label is whether or not this node is positive, negative, or neither (implying a branch)
        self.label = None

        # This is used during classification. If the input value does not exist, use this value to branch
        self.most_common_value = None

        # Child nodes to branch to
        self.children = dict()

        # The attribute that will be tested at this node
        self.attribute = None

        # Label used if there was not a most common value for the attribute
        self.fallback_label = None

        if forced_label:
            self.label = forced_label
            if self.label == b'True':
                metrics.true_nodes += 1
            else:
                metrics.false_nodes += 1
            return

        # Check to see if all positive/negative
        positive_count = 0
        negative_count = 0
        for example in data:
            if example[target_attribute] == target_value:
                positive_count += 1
            elif example[target_attribute] != target_value:
                negative_count += 1
            if negative_count and positive_count:
                break

        if negative_count == 0:
            # All are positive
            self.label = target_value
            if self.label == b'True':
                metrics.true_nodes += 1
            else:
                metrics.false_nodes += 1
            return
        elif positive_count == 0:
            # All are negative
            self.label = negative_value
            if self.label == b'True':
                metrics.true_nodes += 1
            else:
                metrics.false_nodes += 1
            return

        # If no more attributes to check, return with label <= mode(data[target_attribute])
        if not attributes:
            # TODO: Verify that the ? is the value we don't care about everywhere
            self.label = self._mode(data, target_attribute, [b'?'])
            if self.label == b'True':
                metrics.true_nodes += 1
            else:
                metrics.false_nodes += 1
            return

        # Get the best attribute
        best_attribute = self._get_best_attribute(data, attributes, target_attribute, attributes_map)
        self.attribute = best_attribute

        decoded_attribute_options = attributes_map[best_attribute][1]
        # Need to encode here
        attribute_options = [option.encode() for option in decoded_attribute_options]

        # Started drinking here. Verify below later...
        option_map = dict()
        possible_attrs = attributes_map[best_attribute][1]
        for possible in possible_attrs:
            # TODO: Clean up the encoding bullshit
            option_map[possible.encode()] = []

        # TODO: REMOVE THIS IF IT DOESNT WORK
        # Used for switching out missing values
        most_common_val_dict = dict()
        for attribute in attributes:
            most_common_val_dict[attribute] = ID3TreeNode._mode(data, attribute, [b'?'])

        for example in data:
            try:
                # TODO: Should probably fix the bug here
                if example[best_attribute] in attribute_options:
                    option_map[example[best_attribute]].append(example)
                elif example[best_attribute] == b'?':
                    option_map[most_common_val_dict[best_attribute]].append(example)
            except:
                pass

        # Find the most common attribute value at this point
        self.most_common_value = self._mode(data, best_attribute, [b'?'])
        if not self.most_common_value:
            # Save our best guess at this node
            self.fallback_label = self._mode(data, target_attribute, [b'?'])

        # Decide whether or not to stop splitting
        # TODO: should we stop here, or create a node for each child?
        stop_splitting = ID3TreeNode._should_stop_splitting(data, best_attribute, attribute_options, target_attribute, target_value, confidence)
        if stop_splitting:
            self.label = ID3TreeNode._mode(data, target_attribute, [b'?'])
            if self.label == b'True':
                metrics.true_nodes += 1
            else:
                metrics.false_nodes += 1
            return

        # Call this a condition node at this point
        metrics.condition_nodes += 1

        # Remove the best attributes from attributes list
        attributes.remove(best_attribute)
        for attr_val, segment_data in option_map.items():
            if not segment_data:
                # TODO: Definitely double check this
                self.children[attr_val] = ID3TreeNode(data, 'Class', b'True', b'False', attributes, attributes_map, confidence, metrics, self._mode(data, target_attribute, [b'?']))
            else:
                self.children[attr_val] = ID3TreeNode(segment_data, 'Class', b'True', b'False', attributes, attributes_map, confidence, metrics)

    def classify(self, input, print_threshold = 0):
        if self.label:
            return self.label
        else:
            if input[self.attribute] == b'?':
                if not self.most_common_value:
                    return self.fallback_label
                value = self.most_common_value
                next_node = self.children[value]
            else:
                value = input[self.attribute]
                next_node = self.children[value]

            if print_threshold > 0:
                print("Attribute: %s; Value: %s" % (self.attribute, value))
            return next_node.classify(input, print_threshold - 1)

    """
    Helper Methods
    """
    @staticmethod
    def _mode(data, target_attribute, disregard_vals):
        counts = dict()
        for row in data:
            val = row[target_attribute]
            if val and val not in disregard_vals:
                if val in counts:
                    counts[val] = counts[val] + 1
                else:
                    counts[val] = 1
        max_count = 0
        ret_val = None
        for val, count in counts.items():
            if count > max_count:
                ret_val = val
                max_count = count
        return ret_val

    @staticmethod
    def _entropy(data, target_attribute):
        counts = dict()
        total = 0.0
        for row in data:
            total += 1
            val = row[target_attribute]
            if val in counts:
                counts[val] = counts[val] + 1.0
            else:
                counts[val] = 1.0

        # Calculate the summation and return
        return sum([-(count/total) * math.log2(count/total) for count in counts.values()])

    @staticmethod
    def _gain_ratio(data, target_attribute, current_attribute, attribute_options, most_common):
        total_entropy = ID3TreeNode._entropy(data, target_attribute)

        # Partition data into discrete value-based buckets
        # TODO: Do non-existent values need to be considered here?
        value_buckets = dict()

        # TODO: Remove redundant checks
        for row in data:
            val = row[current_attribute]
            if val == b'?':
                val = most_common
            if val not in value_buckets:
                value_buckets[val] = []
            value_buckets[val].append(row)

        total_count = len(data)
        partial_entropies = sum([(len(bucket)/total_count) * ID3TreeNode._entropy(bucket, target_attribute)
                                 for bucket in value_buckets.values()])

        gain = total_entropy - partial_entropies

        split_in_info = sum([-(len(bucket)/total_count) * math.log2(len(bucket)/total_count)
                             for bucket in value_buckets.values()])

        if split_in_info == 0:
            return 0

        return gain / split_in_info

    @staticmethod
    def _get_best_attribute(data, attributes, target_attribute, attributes_map):
        # Get the GainRatio for each attribute
        gain_ratio_dict = dict()

        # Used for switching out missing values
        most_common_val_dict = dict()
        for attribute in attributes:
            most_common_val_dict[attribute] = ID3TreeNode._mode(data, attribute, [b'?'])

        for attribute in attributes:
            # Make sure we don't actually use "Class" for the gain ratio
            attribute_options = attributes_map[attribute][1]
            if attribute != target_attribute:
                gain_ratio_dict[attribute] = ID3TreeNode._gain_ratio(data, target_attribute, attribute, attribute_options, most_common_val_dict[attribute])

        return max(gain_ratio_dict.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _should_stop_splitting(data, attribute, attribute_vals, target_attr, target_val, confidence):
        positive_count = len([row for row in data if row[target_attr] == target_val])
        negative_count = len(data) - positive_count

        num_attrs = 0
        total = 0
        for val in attribute_vals:
            num_attrs += 1
            rows_with_value = [row for row in data if row[attribute] == val]
            local_positive_count = len([row for row in rows_with_value if row[target_attr] == target_val])
            local_negative_count = len(rows_with_value) - local_positive_count
            expected_positive = len(rows_with_value) * positive_count / len(data)
            expected_negative = len(rows_with_value) * negative_count / len(data)

            positive_part = ((local_positive_count - expected_positive) ** 2) / expected_positive if expected_positive else 0
            negative_part = ((local_negative_count - expected_negative) ** 2) / expected_negative if expected_negative else 0
            total = total + positive_part + negative_part

        # Do the chi2 test now
        # TODO: Verify that these numbers actually work
        chi_val = chi2.isf(1 - confidence, num_attrs - 1)

        # Total > chi_val means keep splitting
        return total < chi_val




if __name__ == '__main__':
    # Set logging level to 0, so info and debug gets output
    logging.root.level = 0
    sys.exit(main())
