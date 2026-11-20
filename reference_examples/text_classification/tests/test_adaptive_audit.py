"""Independent small-population checks and exact logged proposal regressions."""
import copy
import itertools
import json
import math
import random
import sys
from contextlib import redirect_stdout
from unittest.mock import patch
from types import SimpleNamespace
import io
import tempfile
import unittest
from pathlib import Path

import adaptive_evolution as e
from adaptive_selection import draw, estimate, make_plan, priorities, relationships
from proposal_workspace import accept_proposal, install_prepared_sources, preflight_candidate_imports


class SubmissionRegressionTests(unittest.TestCase):
    def layout(self, tmp):
        root=Path(tmp)/"run"; code=root/"code"; work=Path(tmp)/"workspace"
        (code/"agents").mkdir(parents=True);(work/"agents").mkdir(parents=True)
        return root,code,work

    def test_exact_iteration4_failure_is_recovered_without_editing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,code,work=self.layout(tmp)
            names=['progressive_threshold','query_adaptive','hybrid_threshold']
            candidates=[]
            hashes={}
            for name in names:
                filename=f'agents/iter0004_{name}.py'
                (work/filename).write_text('VALUE = 42\n')
                hashes[name]=e.file_hash(work/filename)
                candidates.append({'name':name,'file':filename})
            pending=work/'pending_eval.json';e.save_json(pending,{'iteration':4,'candidates':candidates})
            result=accept_proposal(root,code,work,{},pending,4,e.file_hash,e.save_json)
            install_prepared_sources(root,code,result['candidates'],e.file_hash,e.event)
            for row,old in zip(result['candidates'],names):
                self.assertEqual(row['name'],'iter0004_'+old)
                self.assertEqual(row['source_hash'],hashes[old])
                self.assertEqual(row['original_submission']['name'],old)
                self.assertEqual(e.file_hash(code/row['file']),hashes[old])
            self.assertEqual(json.loads(pending.read_text())['candidates'],candidates)

    def test_reversed_prefix_metadata_resolves_to_only_existing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,code,work=self.layout(tmp)
            (work/'agents/iter0004_new.py').write_text('# source')
            pending=work/'pending_eval.json'
            e.save_json(pending,{'iteration':4,'candidates':[{'name':'iter0004_new','file':'agents/new.py'}]})
            result=accept_proposal(root,code,work,{},pending,4,e.file_hash,e.save_json)
            self.assertEqual(result['candidates'][0]['file'],'agents/iter0004_new.py')

    def test_ambiguous_unrelated_and_wrong_iteration_names_still_rejected(self):
        for name,filename,both in [('new','iter0004_new',True),('new','unrelated',False),('new','iter0003_new',False)]:
            with self.subTest(filename=filename),tempfile.TemporaryDirectory() as tmp:
                root,code,work=self.layout(tmp)
                (work/'agents'/f'{filename}.py').write_text('# source')
                if both:(work/'agents/new.py').write_text('# other')
                pending=work/'pending_eval.json'
                e.save_json(pending,{'iteration':4,'candidates':[{'name':name,'file':f'agents/{filename}.py'}]})
                with self.assertRaises(ValueError):accept_proposal(root,code,work,{},pending,4,e.file_hash,e.save_json)
                self.assertFalse((root/'iteration-0004.json').exists())

    def test_normalization_cannot_create_duplicate_names_or_overwrite_agents(self):
        for duplicate in (True,False):
            with tempfile.TemporaryDirectory() as tmp:
                root,code,work=self.layout(tmp)
                (work/'agents/iter0004_new.py').write_text('# source')
                candidates=[{'name':'new','file':'agents/iter0004_new.py'}]
                if duplicate:candidates.append({'name':'iter0004_new','file':'agents/iter0004_new.py'})
                else:(code/'agents/iter0004_new.py').write_text('# original')
                pending=work/'pending_eval.json';e.save_json(pending,{'iteration':4,'candidates':candidates})
                with self.assertRaises(ValueError):accept_proposal(root,code,work,{},pending,4,e.file_hash,e.save_json)
                self.assertFalse((root/'iteration-0004.json').exists())


    def test_import_side_effects_cannot_change_source_before_publication(self):
        for old_file in (True,False):
            with tempfile.TemporaryDirectory() as tmp:
                root,code,work=self.layout(tmp)
                candidate=work/'agents/new.py';candidate.write_text('# new')
                original=work/'agents/old.py';original.write_text('# old')
                existing={'agents/old.py':e.file_hash(original)}
                pending=work/'pending_eval.json';e.save_json(pending,{'iteration':4,'candidates':[{'name':'new'}]})
                def mutate(rows):
                    (original if old_file else candidate).write_text('# changed during import')
                with self.assertRaisesRegex(ValueError,'Preflight changed'):
                    accept_proposal(root,code,work,existing,pending,4,e.file_hash,e.save_json,preflight=mutate)
                self.assertFalse((root/'iteration-0004.json').exists())

    @unittest.skipUnless(sys.platform=='linux','Linux filesystem sandbox')
    def test_import_failure_and_timeout_are_detected_before_evaluation(self):
        for source,timeout in [('import deliberately_missing_module_abcdefgh',5),('import time; time.sleep(10)',0.5)]:
            with tempfile.TemporaryDirectory() as tmp:
                workspace=Path(tmp)/'workspace';work=workspace/'code/text_classification'
                (work/'agents').mkdir(parents=True)
                (work/'agents/new.py').write_text(source)
                root=Path(tmp)/'run';root.mkdir()
                with self.assertRaisesRegex(ValueError,'import preflight failed'):
                    preflight_candidate_imports(workspace,work,[{'name':'new'}],root,e.event,timeout=timeout)
                events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
                self.assertEqual(events[-1]['stage'],'candidate_preflight_failed')
                self.assertFalse((root/'iteration-0004.json').exists())

class CoreAuditTests(unittest.TestCase):
    def history(self):
        return {str(i):{0:a,1:a,2:1-a,3:b} for i,(a,b) in enumerate([(0,0),(0,1),(1,0),(1,1),(0,1),(1,0)])}

    def settings(self):
        return dict(policy='adaptive-redundancy',fraction=0.5,alpha=2,exploration=0.1,
                    min_overlap=3,shrinkage=5,probability_draws=12000,seed=42)

    def test_priorities_and_supported_correlations_match_hand_calculations(self):
        weights,anchors=priorities({'a':{0:1,1:0},'b':{0:1}},2)
        self.assertAlmostEqual(weights[0],0.1/math.sqrt(2))
        self.assertAlmostEqual(weights[1],0.6)
        self.assertEqual(anchors,{0:1,1:0})
        pairs,_=relationships(self.history(),4,3,5)
        self.assertAlmostEqual(pairs[(0,1)],6/11)
        self.assertNotIn((0,2),pairs)  # Perfect negative correlation is not penalized.
        self.assertEqual(relationships(self.history(),4,7,5)[0],{})

    def test_exact_order_enumeration_matches_draw_and_monte_carlo_inclusion(self):
        history=self.history();settings=self.settings()
        weights,_=priorities(history,4);pairs,_=relationships(history,4,3,5)
        inclusion={i:0.0 for i in range(4)};total=0
        # Independent reference implementation of the proposal equations.
        def probabilities(selected):
            adjusted={i:weights[i]/(1+2*max([pairs.get(tuple(sorted((i,j))),0) for j in selected] or [0]))
                      for i in range(4) if i not in selected}
            return {i:0.9*v/sum(adjusted.values())+0.1/len(adjusted) for i,v in adjusted.items()}
        class Numbers:
            def __init__(self,values):self.values=iter(values)
            def random(self):return next(self.values)
        for order in itertools.permutations(range(4),2):
            probability=1;selected=[];thresholds=[]
            for chosen in order:
                probs=probabilities(selected)
                probability*=probs[chosen]
                thresholds.append(sum(p for i,p in probs.items() if i<chosen)+probs[chosen]/2)
                selected.append(chosen)
            actual,_=draw(weights,pairs,2,2,0.1,Numbers(thresholds))
            self.assertEqual(actual,list(order))
            total+=probability
            for i in order:inclusion[i]+=probability
        self.assertAlmostEqual(total,1)
        self.assertAlmostEqual(sum(inclusion.values()),2)
        before=copy.deepcopy(history)
        plan=make_plan(history,4,'audit',settings)
        self.assertEqual(history,before)
        self.assertEqual(plan,make_plan(history,4,'audit',settings))
        for i,exact in inclusion.items():
            self.assertAlmostEqual(plan['inclusion_probabilities'][i],exact,delta=0.025)

    def test_estimators_use_inclusion_not_step_probabilities(self):
        plan={'selected':[0,2],'pool_size':4,'inclusion_probabilities':{'0':0.2,'1':0.4,'2':0.8,'3':0.6},
              'anchors':{'0':0.1,'1':0.2,'2':0.3,'3':0.4}}
        outcomes={0:1.,2:0.}
        hajek=estimate(outcomes,plan,'hajek')
        self.assertAlmostEqual(hajek['estimated_accuracy'],0.8)  # 5 / (5 + 1.25)
        difference=estimate(outcomes,plan,'difference')
        expected=(1.0+(1-0.1)/0.2+(0-0.3)/0.8)/4
        self.assertAlmostEqual(difference['unclipped_estimate'],expected)
        self.assertEqual(difference['estimated_accuracy'],min(1,expected))
        with self.assertRaises(ValueError):estimate({0:1},plan,'hajek')

    def test_full_pool_and_zero_penalty_controls(self):
        settings=self.settings();settings['probability_draws']=100
        full=make_plan(self.history(),4,'full',{**settings,'policy':'full'})
        self.assertEqual(full['selected'],[0,1,2,3])
        self.assertTrue(all(p==1 for p in full['inclusion_probabilities'].values()))
        self.assertEqual(estimate({0:1,1:0,2:1,3:0},full,'hajek')['estimated_accuracy'],0.5)
        control=make_plan(self.history(),4,'control',{**settings,'alpha':0})
        self.assertEqual(control['penalized_steps'],0)
        self.assertFalse(control['selection_changed_vs_alpha_zero'])

class FinalSelectionAuditTests(unittest.TestCase):
    def test_noisy_shortlist_cannot_discard_better_measured_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);code=root/'code';(code/'agents').mkdir(parents=True)
            cfg={'datasets':['Symptom2Disease'],'dataset':{'num_train':4,'num_val':10,'num_test':100}}
            state={'manifest':{'config':cfg},'records':[]}
            for name,estimated,full in [('baseline',0.6,True),('a',0.9,False),('b',0.8,False),('c',0.7,False)]:
                source=code/'agents'/f'{name}.py';source.write_text('# '+name)
                memory=root/f'{name}-memory.json';memory.write_text('{}')
                state['records'].append({'name':name,'candidate_id':name,'source_hash':e.file_hash(source),
                    'iteration':0 if full else 1,'status':'complete','outcomes':{},'memory':str(memory),
                    'score':{'subset_accuracy':estimated,'estimated_accuracy':estimated,
                             'approximate_se':0 if full else 0.15,'measured_full':full}})
            calls=[]
            def evaluate(root,code,cfg,args,name,indices,phase,cid,load_memory=None):
                calls.append((name,phase))
                return {'accuracy':{'a':0.1,'b':0.2,'c':0.3,'baseline':0.55}[name],
                        'total':100 if phase=='test' else 10},load_memory
            with patch.object(e,'evaluate',side_effect=evaluate),redirect_stdout(io.StringIO()):
                e.finalize(root,code,cfg,SimpleNamespace(),state)
            final=e.read_json(root/'finalized.json')
            self.assertEqual(final['shortlist'],['a','b','c'])
            self.assertEqual(final['winner'],'baseline')
            self.assertIn(['baseline',0.6],final['confirmation_scores'])
            self.assertEqual([c for c in calls if c[1]=='test'],[('baseline','test')])
            self.assertNotIn(('baseline','confirmation'),calls)
