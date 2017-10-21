import sys
import logging
import math

from scipy.io import arff


def main():
    logging.info("---Starting application---")
    #data, meta = arff.loadarff("./test_data/tennis.arff")
    data, meta = arff.loadarff("./test_data/training_subsetD.arff")

    logging.info("---Data was loaded successfully---")

    #attributes = [attr.encode() for attr in meta._attrnames]

    #test = dict()
    #for row in data:
    #    test[row['Num Pin Dot Pattern Views']] = 0
    root = ID3TreeNode(data, 'Class', b'True', b'False', meta._attrnames, meta._attributes)
    logging.info("---ID3 decision tree was created successfully---")

    total = 0
    correct = 0
    for row in data:
        prediction = root.classify(row)
        actual = row['Class']
        total += 1
        correct = correct + 1 if prediction == actual else correct
        print("Prediction: %s; Actual: %s" % (prediction, actual))

    print("Correct: %d" % correct)
    print("Total: %d" % total)
    print("Percent: %s" % str(correct / total))



class ID3TreeNode:
    true_nodes = 0
    false_nodes = 0
    def __init__(self, data, target_attribute, target_value, negative_value, attributes, attributes_map, forced_label = None):
        # Label is whether or not this node is positive, negative, or neither (implying a branch)
        self.label = None

        # This is used during classification. If the input value does not exist, use this value to branch
        self.most_common_value = None

        # Child nodes to branch to
        self.children = dict()

        # The attribute that will be tested at this node
        self.attribute = None

        if forced_label:
            self.label = forced_label
            if self.label == b'True':
                ID3TreeNode.true_nodes += 1
            else:
                ID3TreeNode.false_nodes += 1
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
                ID3TreeNode.true_nodes += 1
            else:
                ID3TreeNode.false_nodes += 1
            return
        elif positive_count == 0:
            # All are negative
            self.label = negative_value
            if self.label == b'True':
                ID3TreeNode.true_nodes += 1
            else:
                ID3TreeNode.false_nodes += 1
            return

        # If no more attributes to check, return with label <= mode(data[target_attribute])
        if not attributes:
            # TODO: Verify that the ? is the value we don't care about everywhere
            self.label = self._mode(data, target_attribute, [b'?'])
            if self.label == b'True':
                ID3TreeNode.true_nodes += 1
            else:
                ID3TreeNode.false_nodes += 1
            return

        # Get the best attribute
        best_attribute = self._get_best_attribute(data, attributes, target_attribute, attributes_map)
        self.attribute = best_attribute

        decoded_attribute_options = attributes_map[best_attribute][1]
        # Need to encode here
        attribute_options = (option.encode() for option in decoded_attribute_options)

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
            if example[best_attribute] in attribute_options:
                option_map[example[best_attribute]].append(example)
            elif example[best_attribute] == b'?':
                option_map[most_common_val_dict[best_attribute]].append(example)

        # Find the most common attribute value at this point
        self.most_common_value = self._mode(data, best_attribute, [b'?'])

        # Remove the best attributes from attributes list
        attributes.remove(best_attribute)
        for attr_val, segment_data in option_map.items():
            if not segment_data:
                #print('%s --> %s' % (self.attribute, attr_val))
                # TODO: Definitely double check this
                self.children[attr_val] = ID3TreeNode(data, 'Class', b'True', b'False', attributes, attributes_map, self._mode(data, target_attribute, [b'?']))
            else:
                #print('%s --> %s' % (self.attribute, attr_val))
                self.children[attr_val] = ID3TreeNode(segment_data, 'Class', b'True', b'False', attributes, attributes_map)

    def classify(self, input):
        if self.label:
            return self.label
        else:
            try:
                attr_val = input[self.attribute]
            except:
                # TODO: Remove. This hasn't been hit again
                print("Das ist nicht gud")
            if input[self.attribute] == b'?':
                next_node = self.children[self.most_common_value]
            else:
                try:
                    next_node = self.children[input[self.attribute]]
                except:
                    print("Child doesn't exist")
            return next_node.classify(input)

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

        # Need to prefill with all possible options
        # TODO: More encode bullshit
        #for option in attribute_options:
        #    value_buckets[option.encode()] = []

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

        #sum_arr = []
        #for bucket in value_buckets.values():
        #    try:
        #        split_val = (len(bucket)/total_count) * math.log2(len(bucket)/total_count)
        #    except:
        #        print("huh?")
        #    sum_arr.append(split_val)
        #split_in_info = -sum(sum_arr)

        if split_in_info == 0:
            print(len(value_buckets.values()))
            print("Gain: %d" % gain)
            print("Total count: %d" % total_count)
            for bucket in value_buckets.values():
                print("Bucket length: %d" % len(bucket))
            logging.warning("Split in info is 0")
            logging.warning("Attribute used: %s" % current_attribute)
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







if __name__ == '__main__':
    # Set logging level to 0, so info and debug gets output
    logging.root.level = 0
    sys.exit(main())
