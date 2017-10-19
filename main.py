import sys
import logging
import math

from scipy.io import arff


def main():
    logging.info("---Starting application---")
    data, meta = arff.loadarff("./test_data/tennis.arff")

    logging.info("---Data was loaded successfully---")


    #attributes = [attr.encode() for attr in meta._attrnames]
    root = ID3TreeNode(data, 'Class', b'True', b'False', meta._attrnames)
    for row in data:
        val = row['Class']
        for item in row:
            if item != b'?' and item != b'NULL':
                print(item)
    pass


class ID3TreeNode:
    # Label is whether or not this node is positive or negative (could probably be neither)
    label = None

    # Child nodes to branch to
    children = None

    # The attribute that will be tested at this node
    attribute = None

    def __init__(self, data, target_attribute, target_value, negative_value, attributes):
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
            label = target_value
            return
        elif positive_count == 0:
            # All are negative
            label = negative_value
            return

        # If no more attributes to check, return with label <= mode(data[target_attribute])
        if not attributes:
            # TODO: Verify that the ? is the value we don't care about everywhere
            label = self._mode(data, target_attribute, [b'?'])
            return

        # Get the best attribute
        best_attribute = self._get_best_attribute(data, attributes, target_attribute)




    """
    Helper Methods
    """
    @staticmethod
    def _mode(data, target_attribute, disregard_vals):
        counts = dict()
        for row in data:
            val = row[target_attribute]
            if val not in disregard_vals:
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
    def _gain_ratio(data, target_attribute, current_attribute):
        total_entropy = ID3TreeNode._entropy(data, target_attribute)

        # Partition data into discrete value-based buckets
        # TODO: Do non-existent values need to be considered here?
        value_buckets = dict()
        for row in data:
            val = row[current_attribute]
            if val not in value_buckets:
                value_buckets[val] = []
            value_buckets[val].append(row)

        total_count = len(data)
        partial_entropies = sum([(len(bucket)/total_count) * ID3TreeNode._entropy(bucket, target_attribute)
                                 for bucket in value_buckets.values()])

        split_in_info = sum([-(len(bucket)/total_count) * math.log2(len(bucket)/total_count)
                             for bucket in value_buckets.values()])

        return (total_entropy - partial_entropies) / split_in_info

    @staticmethod
    def _get_best_attribute(data, attributes, target_attribute):
        # Get the GainRatio for each attribute
        gain_ratio_dict = dict()
        for attribute in attributes:
            if attribute != target_attribute:
                gain_ratio_dict[attribute] = ID3TreeNode._gain_ratio(data, target_attribute, attribute)

        return max(gain_ratio_dict.items(), key=lambda item: item[1])[0]



        pass






if __name__ == '__main__':
    # Set logging level to 0, so info and debug gets output
    logging.root.level = 0
    sys.exit(main())
