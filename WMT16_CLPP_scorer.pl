#!/usr/bin/perl
#
#  Author: Preslav Nakov
#  
#  Description: Scores the output for the Shared Task
#               on Cross-lingual Pronoun Prediction, held at WMT'16
#
#
#  Last modified: Ferbuary 19, 2016
#  Additions 
#
#
#  Use:
#     perl WMT16_CLPP_scorer.pl <GOLD_FILE> <PREDICTIONS_FILE> <LANGUAGE_PAIR>
#
#  Example use:
#     perl WMT16_CLPP_scorer.pl gold.txt predicted.txt en-fr
#     perl WMT16_CLPP_scorer.pl gold.txt predicted_simple.txt en-fr
#     perl WMT16_CLPP_scorer.pl TEDdev.en-de.data.filtered TEDdev.en-de.none0.0.predictions.txt en-de
#     perl WMT16_CLPP_scorer.pl TEDdev.de-en.data.filtered TEDdev.de-en.none0.0.predictions.txt de-en
#     perl WMT16_CLPP_scorer.pl TEDdev.en-fr.data.filtered TEDdev.en-fr.none0.0.predictions.txt en-fr
#
#  Description:
#
#     The scorer calculates and outputs the following statistics:
#        (1) confusion matrix, which shows
#			- the count for each gold/predicted pair
#           - the sums for each row/column: -SUM-
#        (2) accuracy
#        (3) precision (P), recall (R), and F1-score for each label
#        (4) micro-averaged P, R, F1 (note that in our single-class classification problem, micro-P=R=F1=Acc)
#        (5) macro-averaged P, R, F1
#
#     The official score is the macro-averaged Recall
#
#

use warnings;
use strict;
use utf8;


###################
###   GLOBALS   ###
###################

my @allLabels_En2Fr    = ('  ce ', 'elle ', 'elles', '  il ', ' ils ', 'cela ', '  on ', 'OTHER');
my %labelMapping_En2Fr = (
	'ce'=>'  ce ', 'c\''=>'  ce ',
    'elle'=>'elle ',
    'elles'=>'elles',
    'il'=>'  il ',
    'ils'=>' ils ',	
	'cela'=>'cela ', 'ça'=>'cela ', 'ca'=>'cela ', 'ç\''=>'cela ',
	'on'=>'  on ',
	'NONE'=>'OTHER',
	'OTHER'=>'OTHER');

my @allLabels_Fr2En    = ('  he ', ' she ', '  it ', 'they ', 'this ', 'these', 'there', 'OTHER');
my %labelMapping_Fr2En = (
    'he'=>'  he ',
    'she'=>' she ',
    'it'=>'  it ',
    'they'=>'they ',
    'this'=>'this ', 'that'=>'this ',
    'these'=>'these', 'those'=>'these',
    'there'=>'there',
	'NONE'=>'OTHER',
	'OTHER'=>'OTHER');

my @allLabels_En2De    = ('  er ', ' sie ', '  es ', ' man ', 'OTHER');
my %labelMapping_En2De = (
    'er'=>'  er ',
    'sie'=>' sie ',
    'es'=>'  es ',
    'man'=>' man ',
	'NONE'=>'OTHER',
	'OTHER'=>'OTHER');

my @allLabels_De2En    = ('  he ', ' she ', '  it ', 'they ', ' you ', 'this ', 'these', 'there', 'OTHER');
my %labelMapping_De2En = (
    'he'=>'  he ',
    'she'=>' she ',
    'it'=>'  it ',
    'they'=>'they ',
    'you'=>' you ',
    'this'=>'this ', 'that'=>'this ',
    'these'=>'these', 'those'=>'these',
    'there'=>'there',
	'NONE'=>'OTHER',
	'OTHER'=>'OTHER');

my @allLabels_Es2En    = ('  he ', ' she ', '  it ', 'they ', ' you ', 'there', 'OTHER');
my %labelMapping_Es2En = (
    'he'=>'  he ',
    'she'=>' she ',
    'it'=>'  it ',
    'they'=>'they ',
    'you'=>' you ',
    'there'=>'there',
	'NONE'=>'OTHER',
	'OTHER'=>'OTHER');


my %confMatrix   = ();
my @allLabels    = ();
my %labelMapping = ();


################
###   MAIN   ###
################

### 1. Check parameters
die "Usage: $0 <GOLD_FILE> <PREDICTIONS_FILE> <LANGAUGE_PAIR>\n" if ($#ARGV != 2);
my $GOLD_FILE        = $ARGV[0];
my $PREDICTIONS_FILE = $ARGV[1];
my $LANGUAGE_PAIR    = $ARGV[2];

if ($LANGUAGE_PAIR eq 'en-fr') {
	@allLabels = @allLabels_En2Fr;
	%labelMapping = %labelMapping_En2Fr;
}
elsif ($LANGUAGE_PAIR eq 'fr-en') {
	@allLabels = @allLabels_Fr2En;
	%labelMapping = %labelMapping_Fr2En;
}
elsif ($LANGUAGE_PAIR eq 'en-de') {
	@allLabels = @allLabels_En2De;
	%labelMapping = %labelMapping_En2De;
}
elsif ($LANGUAGE_PAIR eq 'de-en') {
	@allLabels = @allLabels_De2En;
	%labelMapping = %labelMapping_De2En;
}
elsif ($LANGUAGE_PAIR eq 'es-en') {
        @allLabels = @allLabels_Es2En;
        %labelMapping = %labelMapping_Es2En;
}
else {
	die "Undefined language pair: $LANGUAGE_PAIR\n";
}

my $permittedWords = join ('|', keys %labelMapping);


### 2. Open the files
open GOLD, '<:encoding(UTF-8)', $GOLD_FILE or die "Error opening $GOLD_FILE!";
open PREDICTED, '<:encoding(UTF-8)', $PREDICTIONS_FILE or die "Error opening $PREDICTIONS_FILE!";

### 3. Collect the statistics
for (my $lineNo = 1; <GOLD>; $lineNo++) {
	
	# 3.1. Get the GOLD label
	# OTHER	le	There 's just no way of getting it right .	Il est impossible de de REPLACE_7 percevoir correctement .	0-0 1-1 1-3 2-2 3-2 4-2 5-3 5-4 6-6 7-5 8-7 9-8
	die "Line $lineNo: Wrong file format for $GOLD_FILE!" if (!/^([^\t]*)\t[^\t]*\t[^\t]+\t[^\t]+\t[^\t]+$/);
	my $goldLabel = $1;

	# 3.2. Get the PREDICTED label
	# ce	c'	There 's just no way of getting it right .	Il est impossible de de REPLACE_7 percevoir correctement .	0-0 1-1 1-3 2-2 3-2 4-2 5-3 5-4 6-6 7-5 8-7 9-8
	die "Line $lineNo: The file $PREDICTIONS_FILE is shorter!" if (!($_ = <PREDICTED>));
	die "Line $lineNo: Wrong file format for $PREDICTIONS_FILE!" if (!/^([^\t\n\r]*)/);
	my $predictedLabel = $1;

	# 3.3. Check the file formats
	if ($goldLabel eq '') {
		if ($predictedLabel eq '') {
			next;
		}
		else {
			die "Line $lineNo: The gold label is empty, but the predicted label is not: $predictedLabel";
		}
	}
	elsif ($predictedLabel eq '') {
		die "Line $lineNo: The predicted label is empty, but the gold label is not: $goldLabel";
	}

	die "Line $lineNo: Wrong file format for $GOLD_FILE: the gold label is '$goldLabel'" if ($goldLabel !~ /^($permittedWords)( ($permittedWords))*$/);
	die "Line $lineNo: Wrong file format for $PREDICTIONS_FILE: the predicted label is '$predictedLabel'" if ($predictedLabel !~ /^($permittedWords)( ($permittedWords))*$/);

	my @goldLabels      = split / /, $goldLabel;
	my @predictedLabels = split / /, $predictedLabel;
	die "Line $lineNo: Different number of labels in the gold and in the predictions file." if ($#goldLabels != $#predictedLabels);

	# 3.4. Update the statistics
	for (my $ind = 0; $ind <= $#goldLabels; $ind++) {
		my $gldLabel = $goldLabels[$ind];
		my $prdctdLabel = $predictedLabels[$ind];
		$confMatrix{$labelMapping{$prdctdLabel}}{$labelMapping{$gldLabel}}++;
	}

}

### 4. -grained evaluation
print "\n<<< I.  EVALUATION >>>\n\n";
my ($officialScore, $accuracy) = &evaluate(\@allLabels, \%confMatrix);

### 5. Output the official score
print "<<< II. OFFICIAL SCORE >>>\n";
printf "\nMACRO-averaged R: %6.2f%s", $officialScore, "%\n\n";

### 6. Close the files
close GOLD or die;
close PREDICTED or die;



################
###   SUBS   ###
################

sub evaluate() {
	my ($allLabels, $confMatrix) = @_;

	### 0. Calculate the horizontal and vertical sums
	my %allLabelsProposed = ();
	my %allLabelsAnswer   = ();
	my ($cntCorrect, $cntTotal) = (0, 0);
	foreach my $labelGold (@{$allLabels}) {
		foreach my $labelProposed (@{$allLabels}) {
			$$confMatrix{$labelProposed}{$labelGold} = 0
				if (!defined($$confMatrix{$labelProposed}{$labelGold}));
			$allLabelsProposed{$labelProposed} += $$confMatrix{$labelProposed}{$labelGold};
			$allLabelsAnswer{$labelGold} += $$confMatrix{$labelProposed}{$labelGold};
			$cntTotal += $$confMatrix{$labelProposed}{$labelGold};
		}
		$cntCorrect += $$confMatrix{$labelGold}{$labelGold};
	}

	### 1. Print the confusion matrix heading
	print "Confusion matrix:\n";
	print "       ";
	foreach my $label (@{$allLabels}) {
		printf " %5s", $label;
	}
	print " <-- classified as\n";
	print "       +";
	foreach (@{$allLabels}) {
		print "------";
	}
	print "+ -SUM-\n";

	### 2. Print the rest of the confusion matrix
	my $freqCorrect = 0;
	foreach my $labelGold (@{$allLabels}) {

		### 2.1. Output the short relation label
		printf " %5s |", $labelGold;

		### 2.2. Output a row of the confusion matrix
		foreach my $labelProposed (@{$allLabels}) {
			printf "%5d ", $$confMatrix{$labelProposed}{$labelGold};
		}

		### 2.3. Output the horizontal sums
		printf "| %5d\n", $allLabelsAnswer{$labelGold};
	}
	print "       +";
	foreach (@{$allLabels}) {
		print "------";
	}
	print "+\n";
	
	### 3. Print the vertical sums
	print " -SUM- ";
	foreach my $labelProposed (@{$allLabels}) {
		printf "%5d ", $allLabelsProposed{$labelProposed};
	}
	print "\n\n";

	### 5. Output the accuracy
	my $accuracy = 100.0 * $cntCorrect / $cntTotal;
	printf "%s%d%s%d%s%5.2f%s", 'Accuracy (calculated for the above confusion matrix) = ', $cntCorrect, '/', $cntTotal, ' = ', $accuracy, "\%\n";

	### 8. Output P, R, F1 for each relation
	my ($macroP, $macroR, $macroF1) = (0, 0, 0);
	my ($microCorrect, $microProposed, $microAnswer) = (0, 0, 0);
	print "\nResults for the individual labels:\n";
	foreach my $labelGold (@{$allLabels}) {

		### 8.3. Calculate P/R/F1
		my $P  = (0 == $allLabelsProposed{$labelGold}) ? 0
				: 100.0 * $$confMatrix{$labelGold}{$labelGold} / $allLabelsProposed{$labelGold};
		my $R  = (0 == $allLabelsAnswer{$labelGold}) ? 0
				: 100.0 * $$confMatrix{$labelGold}{$labelGold} / $allLabelsAnswer{$labelGold};
		my $F1 = (0 == $P + $R) ? 0 : 2 * $P * $R / ($P + $R);

		printf "%10s%s%5d%s%5d%s%6.2f", $labelGold,
			" :    P = ", $$confMatrix{$labelGold}{$labelGold}, '/', $allLabelsProposed{$labelGold}, ' = ', $P;

		printf "%s%5d%s%5d%s%6.2f%s%6.2f%s\n", 
		  	 "%     R = ", $$confMatrix{$labelGold}{$labelGold}, '/', $allLabelsAnswer{$labelGold},   ' = ', $R,
			 "%     F1 = ", $F1, '%';

		### 8.5. Accumulate statistics for micro/macro-averaging
		$macroP  += $P;
		$macroR  += $R;
		$macroF1 += $F1;
		$microCorrect += $$confMatrix{$labelGold}{$labelGold};
		$microProposed += $allLabelsProposed{$labelGold};
		$microAnswer += $allLabelsAnswer{$labelGold};
	}

	### 9. Output the micro-averaged P, R, F1
	my $microP  = (0 == $microProposed)    ? 0 : 100.0 * $microCorrect / $microProposed;
	my $microR  = (0 == $microAnswer)      ? 0 : 100.0 * $microCorrect / $microAnswer;
	my $microF1 = (0 == $microP + $microR) ? 0 :   2.0 * $microP * $microR / ($microP + $microR);
	print "\nMicro-averaged result:\n";
	printf "%s%5d%s%5d%s%6.2f%s%5d%s%5d%s%6.2f%s%6.2f%s\n",
		      "P = ", $microCorrect, '/', $microProposed, ' = ', $microP,
		"%     R = ", $microCorrect, '/', $microAnswer, ' = ', $microR,
		"%     F1 = ", $microF1, '%';

	### 10. Output the macro-averaged P, R, F1
	my $distinctLabelsCnt = $#{$allLabels}+1; 

	$macroP  /= $distinctLabelsCnt; # first divide by the number of non-Other categories
	$macroR  /= $distinctLabelsCnt;
	$macroF1 /= $distinctLabelsCnt;
	print "\nMACRO-averaged result:\n";
	printf "%s%6.2f%s%6.2f%s%6.2f%s\n\n\n\n", "P = ", $macroP, "%\tR = ", $macroR, "%\tF1 = ", $macroF1, '%';

	### 11. Return the official score
	return ($macroR, $accuracy);
}
