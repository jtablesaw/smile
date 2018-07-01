/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package tech.tablesaw.api.ml.association;

import it.unimi.dsi.fastutil.ints.IntRBTreeSet;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.NumberColumn;
import smile.association.ARM;
import smile.association.AssociationRule;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.table.TableSlice;
import tech.tablesaw.table.TableSliceGroup;

import java.util.Arrays;
import java.util.List;

/**
 * Association Rule Mining is an unsupervised mining technique related to frequent itemsets
 * <p>
 * Where frequent itemset analysis is concerned only with identifying items that are found together in many baskets,
 * and labeling them with how often they are found. This can be confusing in that there may be some items that are
 * individually very common, and so they appear in the same basket frequently just by chance.
 * <p>
 * Association Rule Mining attempts to identify frequent itemsets that are surprising: That is to say, where the items
 * appear together much more frequently (or less frequently) than one would expect by chance alone
 */
public class AssociationRuleMining {

    private final ARM model;

    public AssociationRuleMining(NumberColumn sets, NumberColumn items, double support) {
        Table temp = Table.create("temp");
        temp.addColumns(sets.copy(), items.copy());
        temp.sortAscendingOn(sets.name(), items.name());

        TableSliceGroup baskets = temp.splitOn(temp.categoricalColumn(0));
        int[][] itemsets = new int[baskets.size()][];
        int basketIndex = 0;
        for (TableSlice basket : baskets) {
            IntRBTreeSet set = new IntRBTreeSet(basket.numberColumn(1).asIntArray());
            int itemIndex = 0;
            itemsets[basketIndex] = new int[set.size()];
            for (int item : set) {
                itemsets[basketIndex][itemIndex] = item;
                itemIndex++;
            }
            basketIndex++;
        }

        this.model = new ARM(itemsets, support);
    }

    public AssociationRuleMining(NumberColumn sets, StringColumn items, double support) {
        Table temp = Table.create("temp");
        temp.addColumns(sets.copy(), items.asNumberColumn());
        temp.sortAscendingOn(sets.name(), items.name());

        TableSliceGroup baskets = temp.splitOn(temp.categoricalColumn(0));
        int[][] itemsets = new int[baskets.size()][];
        int basketIndex = 0;
        for (TableSlice basket : baskets) {
            IntRBTreeSet set = new IntRBTreeSet(basket.numberColumn(1).asIntArray());
            int itemIndex = 0;
            itemsets[basketIndex] = new int[set.size()];
            for (int item : set) {
                itemsets[basketIndex][itemIndex] = item;
                itemIndex++;
            }
            basketIndex++;
        }

        this.model = new ARM(itemsets, support);
    }

    public List<AssociationRule> learn(double confidenceThreshold) {
        return model.learn(confidenceThreshold);
    }

    public List<AssociationRule> interestingRules(double confidenceThreshold,
                                                  double interestThreshold,
                                                  Object2DoubleOpenHashMap<IntRBTreeSet> confidenceMap) {
        List<AssociationRule> rules = model.learn(confidenceThreshold);
        for (AssociationRule rule : rules) {
            double interest = rule.confidence - confidenceMap.getDouble(rule.consequent);
            if (Math.abs(interest) < interestThreshold) {
                rules.remove(rule);
            }
        }
        return rules;
    }

    public Table interest(double confidenceThreshold,
                          double interestThreshold,
                          Object2DoubleOpenHashMap<IntRBTreeSet> confidenceMap) {

        Table interestTable = Table.create("Interest");
        interestTable.addColumns(
                StringColumn.create("Antecedent"),
                StringColumn.create("Consequent"),
                DoubleColumn.create("Confidence"),
                DoubleColumn.create("Interest"));

        List<AssociationRule> rules = model.learn(confidenceThreshold);

        for (AssociationRule rule : rules) {
            double interest = rule.confidence - confidenceMap.getDouble(new IntRBTreeSet(rule.consequent));
            if (Math.abs(interest) > interestThreshold) {
                interestTable.stringColumn(0).appendCell(Arrays.toString(rule.antecedent));
                interestTable.stringColumn(1).appendCell(Arrays.toString(rule.consequent));
                interestTable.numberColumn(2).append(rule.confidence);
                interestTable.numberColumn(3).append(interest);
            }
        }
        return interestTable;
    }
}
