/* ============================================================================
   Concept Pairs — a jsPsych study for the Kombine combinatorial-creativity
   benchmark. Participants make an ASSOCIATION, ANALOGY, or BLEND between
   arbitrary entities, then rate surprise / emergence / confidence.

   jsPsych 8. Plugins: instructions, survey-html-form, html-button-response.
   ========================================================================== */

/* ---- config you may want to change ------------------------------------- */
var N_PER_TASK = 2;                      // trials of each of the three tasks
var TASK_ORDER = "interleaved";          // "interleaved" or "blocked"
var PROLIFIC_COMPLETION_URL = "";        // e.g. "https://app.prolific.com/submissions/complete?cc=XXXX"
/* DataPipe (https://pipe.jspsych.org) — set an experiment id to save to OSF.
   Leave "" to instead offer a local JSON download at the end.               */
var DATAPIPE_EXPERIMENT_ID = "";
/* ------------------------------------------------------------------------ */

/* Capture Prolific / URL params so they ride along in the data. */
function param(name){
  var v = new URLSearchParams(window.location.search).get(name);
  return v === null ? "" : v;
}
var subject = {
  prolific_pid: param("PROLIFIC_PID"),
  study_id:     param("STUDY_ID"),
  session_id:   param("SESSION_ID")
};

var jsPsych = initJsPsych({
  display_element: undefined,
  on_finish: function(){ saveData(); }
});
jsPsych.data.addProperties(subject);

/* ---------------------------------------------------------------- content */
var POOL = ["a violin","Antarctica","the stock market","a lighthouse","photosynthesis","jazz",
  "the immune system","origami","a suspension bridge","coffee","the printing press","chess",
  "a coral reef","Morse code","a glacier","the Silk Road","a beehive","gravity","a metronome",
  "Braille","sonar","the abacus","a whirlpool","the water cycle","a volcano","the postal service",
  "a spider's web","the tides","a cathedral","fireflies","a compass","penicillin"];

var BLEND = ["virus","cloud","stream","web","memory","branch","current","bug","mouse","window",
  "net","wave","field","key","port","spring","cell","crane","bark","pitch","root","bridge"];

var EYE = {
  association: "Association — build a chain",
  analogy:     "Analogy — find the parallel",
  blend:       "Blend — a word's second life"
};

var EXAMPLE = {
  association: "<b>rubber</b> &rarr; bounces because it's elastic &rarr; elasticity stores energy &rarr; a spring stores energy &rarr; <b>a pogo stick</b>. Each step is a true link; together they carry you across.",
  analogy:     "A <b>glacier</b> is to a mountain valley as <b>a printing press</b> is to literacy &mdash; each slowly and unstoppably reshapes the ground beneath it and leaves a lasting imprint.",
  blend:       "<b>stream</b> &mdash; water in a channel, and data to your screen. Both have a <b>source</b>, a <b>current</b> that runs fast or slow, and can be <b>dammed</b> / <b>buffered</b>. The second sense borrows the shape of the first."
};

/* rating scale words */
var SURP = ["","obvious","close","related","somewhat","distant","far apart","total strangers"];
var NOV  = ["","nothing","barely","slight","a little","a new angle","a real insight","a new idea"];
var CONF = ["","a stretch","shaky","tentative","unsure","fairly solid","confident","rock solid"];

/* ------------------------------------------------------------- sampling */
function shuffle(a){ for(var i=a.length-1;i>0;i--){ var j=Math.floor(Math.random()*(i+1)); var t=a[i]; a[i]=a[j]; a[j]=t; } return a; }

function buildTrials(){
  var kinds = [];
  ["association","analogy","blend"].forEach(function(k){ for(var i=0;i<N_PER_TASK;i++) kinds.push(k); });
  if (TASK_ORDER === "interleaved"){
    // round-robin so the same task never repeats back-to-back
    var buckets = {association:[],analogy:[],blend:[]};
    kinds.forEach(function(k){ buckets[k].push(k); });
    kinds = [];
    for(var r=0;r<N_PER_TASK;r++){ ["association","analogy","blend"].forEach(function(k){ if(buckets[k].length) kinds.push(buckets[k].pop()); }); }
  } else {
    kinds = shuffle(kinds);
  }
  var pairPool = shuffle(POOL.slice());
  var blendPool = shuffle(BLEND.slice());
  var out = [];
  kinds.forEach(function(k){
    if (k === "blend"){
      out.push({ type:k, anchor: blendPool.pop() });
    } else {
      out.push({ type:k, a: pairPool.pop(), b: pairPool.pop() });
    }
  });
  return out;
}
var TRIALS = buildTrials();

/* ---------------------------------------------------------- HTML builders */
function esc(s){ return (s==null?"":String(s)).replace(/&/g,"&amp;").replace(/"/g,"&quot;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

function promptHTML(t){
  if (t.type === "blend"){
    return '<div class="pairline"><span class="chip solo">'+esc(t.anchor)+'</span></div>';
  }
  var tween = t.type === "analogy" ? "&amp;" : "to";
  return '<div class="pairline"><span class="chip a">'+esc(t.a)+'</span>'+
         '<span class="tween">'+tween+'</span>'+
         '<span class="chip b">'+esc(t.b)+'</span></div>';
}

function inputHTML(t){
  if (t.type === "association"){
    return ''+
      '<p class="ask">Lay a chain of true links leading from <b>'+esc(t.a)+'</b> to <b>'+esc(t.b)+'</b>.</p>'+
      '<div class="path" id="pathHost"></div>'+
      '<div class="addstep"><button type="button" id="addStep">+ add a stepping-stone</button></div>';
  }
  if (t.type === "analogy"){
    return ''+
      '<p class="ask">Complete a parallel: each of these relates to something in the <b>same way</b>.</p>'+
      '<div class="analogy">'+
        '<div class="arow"><span class="from a">'+esc(t.a)+'</span>'+
          '<input type="text" name="a_to" placeholder="relates to&hellip; (fill in)"></div>'+
        '<div class="arow"><span class="from b">'+esc(t.b)+'</span>'+
          '<input type="text" name="b_to" placeholder="relates to&hellip; (fill in)"></div>'+
        '<div class="rel-shared"><label class="fld">The shared relationship'+
          '<input type="text" name="shared_relation" required placeholder="&hellip;in what identical way? e.g. protects a fragile thing inside"></label></div>'+
      '</div>';
  }
  // blend
  return ''+
    '<p class="ask">Give <b>'+esc(t.anchor)+'</b> a second, unrelated meaning in a different domain.</p>'+
    '<div class="senses">'+
      '<div class="sense"><span class="stag">Sense 1 &middot; the familiar one</span>'+
        '<textarea name="sense1" rows="2" placeholder="What does &ldquo;'+esc(t.anchor)+'&rdquo; usually mean?"></textarea></div>'+
      '<div class="sense two"><span class="stag">Sense 2 &middot; a different domain</span>'+
        '<textarea name="sense2" rows="2" required placeholder="Invent a second meaning far from the first"></textarea></div>'+
    '</div>'+
    '<label class="fld">What structure do both senses share?'+
      '<input type="text" name="shared_frame" placeholder="e.g. both spread, can be caught, and need defending against"></label>';
}

function ratingsHTML(){
  function slider(name,label,sub,lo,hi){
    return '<div class="rate">'+
      '<div class="rlabel">'+label+' <small>'+sub+'</small></div>'+
      '<output id="o_'+name+'">4</output>'+
      '<input type="range" name="'+name+'" id="r_'+name+'" min="1" max="7" value="4">'+
      '<div class="scaleends"><span>'+lo+'</span><span>'+hi+'</span></div>'+
    '</div>';
  }
  return '<div class="ratings">'+
    slider("surprise","Surprise","How far apart did these ideas feel to start?","obviously related","total strangers")+
    slider("emergence","Emergence","Did your link reveal something neither idea carried alone?","nothing new","a genuinely new idea")+
    slider("confidence","Confidence","How solid / true is the link you made?","a stretch","rock solid")+
  '</div>';
}

function trialHTML(t, i){
  return ''+
    '<div class="prompt-eyebrow"><span class="num">'+("0"+(i+1)).slice(-2)+'</span>'+
      '<span class="eyebrow">'+EYE[t.type]+'</span></div>'+
    '<div class="stage" data-t="'+t.type+'">'+
      promptHTML(t)+
      inputHTML(t)+
      '<details class="example"><summary>See a worked example</summary>'+
        '<div class="body">'+EXAMPLE[t.type]+'</div></details>'+
      ratingsHTML()+
    '</div>';
}

/* --------------------------------------------------------- trial behavior */
function wireTrial(t){
  // live rating labels
  [["surprise",SURP],["emergence",NOV],["confidence",CONF]].forEach(function(p){
    var r = document.getElementById("r_"+p[0]), o = document.getElementById("o_"+p[0]);
    if(!r) return;
    var upd = function(){ o.textContent = r.value + " · " + p[1][+r.value]; };
    r.addEventListener("input", upd); upd();
  });

  if (t.type === "association"){
    var steps = [{}];               // start with one empty step
    var host = document.getElementById("pathHost");
    var draw = function(){
      var h = '<div class="node"><span class="dot a"></span><span class="lbl a">'+esc(t.a)+'</span></div>';
      steps.forEach(function(s,i){
        h += '<div class="rung"><span class="rail"></span><div class="stepwrap">'+
             '<input class="relinput" type="text" name="step_rel_'+i+'" placeholder="&mdash; relation &rarr;">'+
             '<input class="entinput" type="text" name="step_ent_'+i+'"'+(i===0?' required':'')+' placeholder="next concept">'+
             (steps.length>1?'<button type="button" class="xbtn" data-rm="'+i+'" aria-label="remove step">&times;</button>':'')+
             '</div></div>';
      });
      h += '<div class="node"><span class="dot b"></span><span class="lbl b">'+esc(t.b)+'</span></div>';
      host.innerHTML = h;
      host.querySelectorAll("[data-rm]").forEach(function(b){
        b.addEventListener("click", function(){ steps.splice(+this.dataset.rm,1); draw(); });
      });
    };
    draw();
    document.getElementById("addStep").addEventListener("click", function(){ steps.push({}); draw(); });
  }
}

/* Reshape the flat form response into tidy structured data. */
function tidy(t, resp){
  var ratings = { surprise:+resp.surprise, emergence:+resp.emergence, confidence:+resp.confidence };
  var clean = { type:t.type, ratings:ratings };
  if (t.type === "association"){
    clean.a = t.a; clean.b = t.b;
    var path = [];
    Object.keys(resp).forEach(function(k){
      var m = k.match(/^step_ent_(\d+)$/); if(!m) return;
      var i = m[1];
      var ent = (resp["step_ent_"+i]||"").trim(), rel = (resp["step_rel_"+i]||"").trim();
      if (ent || rel) path.push({ index:+i, relation:rel, entity:ent });
    });
    path.sort(function(a,b){ return a.index-b.index; });
    clean.path = path;
  } else if (t.type === "analogy"){
    clean.a = t.a; clean.b = t.b;
    clean.a_to = (resp.a_to||"").trim();
    clean.b_to = (resp.b_to||"").trim();
    clean.shared_relation = (resp.shared_relation||"").trim();
  } else {
    clean.anchor = t.anchor;
    clean.sense1 = (resp.sense1||"").trim();
    clean.sense2 = (resp.sense2||"").trim();
    clean.shared_frame = (resp.shared_frame||"").trim();
  }
  return clean;
}

/* --------------------------------------------------------------- timeline */
var timeline = [];

/* consent */
timeline.push({
  type: jsPsychSurveyHtmlForm,
  button_label: "Begin the study",
  html:
    '<div class="intro">'+
      '<p class="eyebrow">Creativity research · concept combination</p>'+
      '<h1>Connect two ideas that don\'t obviously belong together.</h1>'+
      '<p class="lede">You\'ll see a series of concept pairs, sampled at random. For each, you make a '+
        'creative link between them, then rate how it felt. There are no right answers &mdash; we study '+
        'how people bridge distant ideas. About 8&ndash;10 minutes.</p>'+
      '<label class="consent"><input type="checkbox" name="consent" required> '+
        'I understand these responses are collected for research on creative cognition, stored without '+
        'identifying information, and may appear in aggregate in a publication.</label>'+
    '</div>'
});

/* instructions with one example per task */
timeline.push({
  type: jsPsychInstructions,
  show_clickable_nav: true,
  button_label_previous: "Back",
  button_label_next: "Next",
  pages: [
    '<div class="instr"><h2>Three kinds of link</h2>'+
      '<p>Each screen asks for one of three moves. Here is one example of each &mdash; you\'ll see them again as you go.</p></div>',
    '<div class="instr t-association"><span class="tk">Association</span><h2>Build a chain</h2>'+
      '<p>Lay stepping-stones from one idea to the other. Each step should be a link you believe is true.</p>'+
      '<p class="egline">'+EXAMPLE.association+'</p></div>',
    '<div class="instr t-analogy"><span class="tk">Analogy</span><h2>Find the parallel</h2>'+
      '<p>Say what each idea relates to, so that both relate in the <em>same</em> way.</p>'+
      '<p class="egline">'+EXAMPLE.analogy+'</p></div>',
    '<div class="instr t-blend"><span class="tk">Blend</span><h2>Give a word a second life</h2>'+
      '<p>Invent a second meaning for the word in a completely different domain, then say what the two senses share.</p>'+
      '<p class="egline">'+EXAMPLE.blend+'</p></div>'
  ]
});

/* task trials */
TRIALS.forEach(function(t, i){
  timeline.push({
    type: jsPsychSurveyHtmlForm,
    button_label: (i === TRIALS.length-1 ? "Submit last one" : "Submit & continue"),
    html: trialHTML(t, i),
    data: { phase:"task", task:t.type, trial_number:i+1 },
    on_load: function(){ wireTrial(t); },
    on_finish: function(data){ data.clean = tidy(t, data.response); }
  });
});

/* debrief + data-out */
timeline.push({
  type: jsPsychHtmlButtonResponse,
  choices: ["Finish"],
  stimulus:
    '<div class="intro"><p class="eyebrow">Complete</p>'+
      '<h1>That\'s the set &mdash; thank you.</h1>'+
      '<p class="lede">Your responses are recorded. '+
        (DATAPIPE_EXPERIMENT_ID ? 'They\'ve been saved to the study server.' :
         'Use the button below to download them, or they\'ll be shown on screen.')+'</p>'+
      '<div id="dl"></div></div>',
  on_load: function(){
    if (DATAPIPE_EXPERIMENT_ID) return;
    var b = document.createElement("button");
    b.className = "jspsych-btn"; b.textContent = "Download data (JSON)";
    b.onclick = function(){ jsPsych.data.get().filter({phase:"task"}).localSave("json","concept-pairs.json"); };
    document.getElementById("dl").appendChild(b);
  }
});

/* ------------------------------------------------------------- data saving */
function saveData(){
  if (PROLIFIC_COMPLETION_URL){
    // small delay so any async save can fire first
    setTimeout(function(){ window.location = PROLIFIC_COMPLETION_URL; }, 400);
  } else {
    // dev fallback: dump everything to the page
    jsPsych.data.displayData("json");
  }
}

/* If using DataPipe, load its plugin in index.html and swap saveData() to call
   jsPsychPipe.saveData({ experiment_id: DATAPIPE_EXPERIMENT_ID, ... }). See README. */

jsPsych.run(timeline);
