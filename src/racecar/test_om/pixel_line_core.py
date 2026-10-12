"""Image-space yellow corridor tracking. Input: binary mask + dt only.
No intrinsics, homography, depth, metric path, odometry or simulator pose.
Engineering adaptation of near/far visual steering and temporal lane tracking;
not a reproduction of B-Snake or a full interaction-matrix IBVS controller.
"""
import math,time
import cv2
import numpy as np

PIXEL_DEFAULTS={
    'target_x_px':160.,  # Desired image centre, adjustable for mounting offset.
    'roi_top_ratio':.63, 'row_step_px':4, 'scan_half_height':2,
    'min_segment_px':1, 'max_segment_px':48, 'min_component_px':8,
    'min_component_height_px':10, 'feature_radius_px':16,
    'fragment_gap_px':20, 'fragment_tolerance_px':14.,
    'width_scale_update':.30,
    'blob_aspect_min':1.6, 'blob_fill_max':.70,
    'path_beam_width':32,
    'path_temperature':.04,
    'mask_cross_cost':.10,
    'warmup_frames':5, 'warmup_votes':2,
    'width_ref_y':195., 'width_ref_px':240., 'width_slope_px':4.8,
    'width_update':.06, 'width_gate_ratio':.50,
    'near_y_ratio':.85, 'far_y_ratio':.72,
    'gain_near':.05, 'gain_far':.045, 'gain_heading':.025,
    'control_tau':.07, 'steer_rate':.70, 'max_steer':.18,
    'derivative_gain_sec':.08, 'derivative_tau':.12, 'max_feature_rate':.4,
    'speed_straight':.70, 'speed_corner':.68, 'speed_ramp':.35,
    'blind_bridge_sec':.35, 'history_timeout':.65,
    'min_valid_rows':3, 'morph_kernel':3,
}


class ImageLineController:
    def __init__(self,cfg=None):
        self.cfg=dict(PIXEL_DEFAULTS,**(cfg or {}));c=self.cfg
        self.wref=c['width_ref_px'];self.wslope=c['width_slope_px']
        self.previous=[];self.t=0.;self.last_seen=-100.;self.steer=self.speed=0.
        self.result=None;self.rows=[];self.mask_y=None;self.pixels=0
        self.connectors=[];self.width_samples=0
        self.feature_previous=None;self.feature_derivative=0.
        self.width_scale=1.
        self.edge_history={}
        self.warmup=[];self.armed=False

    def image_feature(self,good,y):
        """Local robust image fit; no whole-V polynomial or ground projection."""
        window=[p for p in good if abs(p['y']-y)<=self.cfg['feature_radius_px']]
        if len(window)<3:window=sorted(good,key=lambda p:abs(p['y']-y))[:min(5,len(good))]
        ys=np.array([p['y'] for p in window],float);xs=np.array([p['center'] for p in window],float)
        slopes=[(xs[j]-xs[i])/(ys[j]-ys[i]) for i in range(len(xs)) for j in range(i+1,len(xs)) if abs(ys[j]-ys[i])>=4]
        slope=float(np.median(slopes)) if slopes else 0.
        return float(np.median(xs-slope*(ys-y)))

    def join_fragments(self,labels,stats,valid_ids):
        """Associate broken pieces in image rows, before deciding left/right.
        Mutual nearest endpoint continuation avoids assigning a new side after
        each segmentation gap. Overlapping row ranges may not be merged.
        """
        c=self.cfg;ends={};parent={k:k for k in valid_ids};ranges={}
        def root(k):
            while parent[k]!=k:k=parent[k]
            return k
        for k in valid_ids:
            y0=int(stats[k,cv2.CC_STAT_TOP]);y1=y0+int(stats[k,cv2.CC_STAT_HEIGHT])-1
            samples=[]
            for y in sorted(set(list(range(y0,min(y0+6,y1+1)))+list(range(max(y0,y1-5),y1+1)))):
                xs=np.flatnonzero(labels[y]==k)
                if len(xs):samples.append((y,float(np.median(xs))))
            top=samples[:min(5,len(samples))];bottom=samples[-min(5,len(samples)):]
            def slope(points):
                return float(np.median([(b[1]-a[1])/(b[0]-a[0]) for i,a in enumerate(points) for b in points[i+1:]])) if len(points)>1 else 0.
            ends[k]=(samples[0],samples[-1],slope(top),slope(bottom));ranges[k]=[(y0,y1)]
        best_down={};best_up={};links=[]
        for a in valid_ids:
            for b in valid_ids:
                if a==b:continue
                topa,bota,sa,sba=ends[a];topb,botb,sb,sbb=ends[b]
                dy=topb[0]-bota[0]
                if not 0<dy<=c['fragment_gap_px']:continue
                err1=abs(topb[1]-bota[1]-sba*dy);err2=abs(topb[1]-bota[1]-sb*dy)
                if max(err1,err2)>c['fragment_tolerance_px']:continue
                score=err1+err2+.15*dy
                links.append((score,a,b))
                if a not in best_down or score<best_down[a][0]:best_down[a]=(score,b)
                if b not in best_up or score<best_up[b][0]:best_up[b]=(score,a)
        for score,a,b in sorted(links):
            if best_down[a][1]!=b or best_up[b][1]!=a:continue
            ra,rb=root(a),root(b)
            if ra==rb:continue
            if any(max(u[0],v[0])<=min(u[1],v[1]) for u in ranges[ra] for v in ranges[rb]):continue
            parent[rb]=ra;ranges[ra]+=ranges[rb]
        lookup=np.arange(len(stats));new_ids=set()
        for k in valid_ids:lookup[k]=root(k);new_ids.add(root(k))
        return lookup[labels],new_ids

    def width_at(self,y):
        # Empirical IMAGE width curve, learned only from valid two-edge pixels.
        return float(np.clip((self.wref+self.wslope*(y-self.cfg['width_ref_y']))*self.width_scale,24,650))

    def predicted_center(self,y):
        if len(self.previous)<2:return self.cfg['target_x_px']
        ys=np.array([p['y'] for p in self.previous]);xs=np.array([p['center'] for p in self.previous])
        return float(np.interp(y,ys,xs))

    def predicted_edge(self,side,y):
        memory=self.edge_history.get(side)
        if memory is not None and self.t-memory[0]<=.45:
            points=memory[1]
            if points[0][0]-8<=y<=points[-1][0]+8:
                return float(np.interp(y,[p[0] for p in points],[p[1] for p in points]))
        return self.predicted_center(y)+(-.5 if side=='left' else .5)*self.width_at(y)

    def reconstruct_path(self,candidates,sides,support):
        """Select one coherent pixel centre path across ALL rows, not greedy
        nearest blobs. Alternative fragment identities survive until later rows
        resolve them. A short spurious near-row cannot anchor the whole path.
        """
        c=self.cfg;beam=[(0.,[],dict(y=239,center=c['target_x_px'],source='anchor'),0.,{})]
        for y in self.rows:
            w=self.width_at(y);prior=self.predicted_center(y);hyp=[];cand=candidates[y]
            for x,k in cand:
                for side in ['left','right']:
                    left=x if side=='left' else x-w;right=x+w if side=='left' else x
                    centre=(left+right)/2
                    cost=.10*min(abs(centre-prior)/w,.8)+.025+.025*(side!=sides[k])+.01*max(0,7-len(support[k]))
                    hyp.append((cost,dict(y=y,left=left,right=right,center=centre,source='single_'+side),[(k,side)],None))
            for i,(a,ka) in enumerate(cand):
                for b,kb in cand[i+1:]:
                    if ka==kb:continue
                    width=b-a
                    if width<24 or abs(width/w-1)>c['width_gate_ratio']:continue
                    if any(len(support[k])<6 or max(support[k])-min(support[k])<24 for k in [ka,kb]):continue
                    centre=(a+b)/2;cost=.10*min(abs(centre-prior)/w,.8)+.04*abs(width/w-1)
                    hyp.append((cost,dict(y=y,left=a,right=b,center=centre,source='double'),[(ka,'left'),(kb,'right')],(y,width,ka,kb)))
            hyp=sorted(hyp,key=lambda v:v[0])[:12];expanded=[];cross_cache={}
            for cost,path,last,slope,identities in beam:
                expanded.append((cost+.14,path+[None],last,slope,identities))
                for local,p,bindings,pair in hyp:
                    if any(k in identities and identities[k]!=side for k,side in bindings):continue
                    transition=0.;new_slope=0.
                    if last is not None:
                        dy=y-last['y'];new_slope=(p['center']-last['center'])/dy
                        deviation=abs(p['center']-last['center']-slope*dy)/w
                        transition=.30*min(deviation,2.)+.006*min(abs(new_slope-slope),10.)
                        key=(last['y'],last['center'],p['center'])
                        if key not in cross_cache:
                            xx=np.linspace(last['center'],p['center'],max(3,int(abs(dy))))
                            yy=np.linspace(last['y'],y,len(xx));inside=(xx>=0)&(xx<320)&(yy>=0)&(yy<240)
                            hit=False
                            if np.any(inside):
                                proximity=self.free_distance[np.rint(yy[inside]).astype(int),np.clip(np.rint(xx[inside]).astype(int),0,319)]
                                hit=bool(np.min(proximity)<2)
                            cross_cache[key]=hit
                        if cross_cache[key]:transition+=c['mask_cross_cost']
                    identity=dict(identities);identity.update(bindings)
                    point=dict(p);point['_pair']=pair
                    expanded.append((cost+local+transition,path+[point],point,float(np.clip(new_slope,-6,6)),identity))
            # Retain different pixel corridors; many nearly identical prefixes
            # must not crowd out a still-plausible alternative boundary identity.
            beam=[];bins={}
            for state in sorted(expanded,key=lambda v:v[0]):
                last=state[2];key=(int(last['center']//20),last['source']) if last is not None else ('none',)
                if bins.get(key,0)>=2:continue
                bins[key]=bins.get(key,0)+1;beam.append(state)
                if len(beam)>=c['path_beam_width']:break
        beam=sorted(beam,key=lambda v:v[0]);path=beam[0][1];pairs=[]
        weights=np.exp(-np.array([v[0]-beam[0][0] for v in beam])/c['path_temperature'])
        for i,p in enumerate(path):
            if p is None:continue
            samples=[(weight,state[1][i]['center']) for weight,state in zip(weights,beam) if state[1][i] is not None]
            total=sum(weight for weight,x in samples)
            centre=sum(weight*x for weight,x in samples)/total
            # Marginalise near-equal interpretations instead of snapping to a
            # different hidden-edge hypothesis on each noisy frame.
            p['consensus_offset']=centre-p['center'];p['center']=centre
        for p in path:
            if p is not None:
                pair=p.pop('_pair')
                if pair is not None:pairs.append(pair)
        return path,pairs

    def observe(self,mask):
        if np.asarray(mask).shape!=(240,320):raise ValueError('Expected 240x320 mask')
        c=self.cfg;binary=(mask>0).astype('uint8');binary[:int(240*c['roi_top_ratio'])]=0
        kernel=np.ones((c['morph_kernel'],)*2,np.uint8)
        binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
        n,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
        widths=np.maximum(stats[:,cv2.CC_STAT_WIDTH],1);heights=np.maximum(stats[:,cv2.CC_STAT_HEIGHT],1)
        aspect=np.maximum(widths/heights,heights/widths)
        fill=stats[:,cv2.CC_STAT_AREA]/(widths*heights)
        valid_ids=set(np.flatnonzero((stats[:,cv2.CC_STAT_AREA]>=c['min_component_px']) &
                      (stats[:,cv2.CC_STAT_HEIGHT]>=c['min_component_height_px']) &
                      ((aspect>=c['blob_aspect_min']) | (fill<c['blob_fill_max']))))-{0}
        binary=np.isin(labels,list(valid_ids)).astype('uint8')
        if valid_ids:labels,valid_ids=self.join_fragments(labels,stats,valid_ids)
        self.free_distance=cv2.distanceTransform((1-binary).astype('uint8'),cv2.DIST_L2,3)
        self.pixels=int(np.count_nonzero(binary))
        yy=np.flatnonzero(np.any(binary,axis=1));self.mask_y=(int(yy[0]),int(yy[-1])) if len(yy) else None
        self.rows=list(range(int(240*c['roi_top_ratio']),238,c['row_step_px']))[::-1]
        candidates={};votes={};support={};self.connectors=[]
        for y in self.rows:
            h=c['scan_half_height'];band=binary[max(0,y-h):min(240,y+h+1)]
            present=np.any(band,axis=0).astype('int16')
            change=np.diff(np.pad(present,(1,1)));starts=np.flatnonzero(change==1);ends=np.flatnonzero(change==-1)-1
            row=[];w=self.width_at(y);cen=self.predicted_center(y)
            for a,b in zip(starts,ends):
                if b-a+1>c['max_segment_px']:
                    self.connectors.append([int(a),int(b),y]);continue
                if b-a+1<c['min_segment_px']:continue
                region=labels[max(0,y-h):min(240,y+h+1),a:b+1];ids=region[region>0]
                if not len(ids):continue
                label=int(np.bincount(ids).argmax())
                if label not in valid_ids:continue
                x=(a+b)/2;row.append((float(x),label))
                support.setdefault(label,set()).add(y)
                dl=abs(x-self.predicted_edge('left',y))/w;dr=abs(x-self.predicted_edge('right',y))/w
                weight=1.0+max(0,y-180)/30
                v=votes.setdefault(label,[0.,0.,0.]);v[0]+=dl*weight;v[1]+=dr*weight;v[2]+=weight
            candidates[y]=row
        # A connected boundary has one identity across rows, even if it crosses x=160.
        sides={k:('left' if v[0]<v[1] else 'right') for k,v in votes.items()}
        observations,pairs=self.reconstruct_path(candidates,sides,support)
        good=[p for p in observations if p is not None]
        if len(good)>=c['min_valid_rows']:
            for side in ['left','right']:
                measured=sorted([(p['y'],p[side]) for p in good if p['source'] in ['double','single_'+side]])
                if len(measured)>=3:self.edge_history[side]=(self.t,measured)
            # Heading changes the apparent image width even when physical lane
            # width is constant. Learn a bounded image-only scale from agreeing
            # TWO-edge rows; never infer scale from a single edge or a blob pair.
            groups={}
            for y,w,a,b in pairs:groups.setdefault((a,b),[]).append((y,w))
            trusted=max(groups.values(),key=len) if groups else []
            if len(trusted)>=6 and max(y for y,w in trusted)-min(y for y,w in trusted)>=24:
                ratios=np.array([w/max(24,self.wref+self.wslope*(y-c['width_ref_y'])) for y,w in trusted])
                mid=float(np.median(ratios));agree=abs(ratios-mid)<.10
                if np.count_nonzero(agree)>=5 and .85<=mid<=1.55:
                    self.width_scale+=(mid-self.width_scale)*c['width_scale_update']
                    for p in good:
                        if p['source']=='single_left':p['right']=p['left']+self.width_at(p['y'])
                        elif p['source']=='single_right':p['left']=p['right']-self.width_at(p['y'])
                        p['center']=(p['left']+p['right'])/2+p.get('consensus_offset',0.)
            # Keep histories indexed by pixel y, never by changing row-slot numbers.
            self.previous=sorted([dict(p,center=self.image_feature(good,p['y'])) for p in good],key=lambda p:p['y'])
            # Freeze width learning during a bend: a folded boundary must not
            # redefine the straight-lane pixel width used to fill a hidden edge.
            center_drift=np.ptp([(p['center']-c['target_x_px'])/self.width_at(p['y']) for p in good])
            if center_drift<.18 and len(trusted)>=6 and max(y for y,w in trusted)-min(y for y,w in trusted)>=24:
                ys,ws=np.asarray(trusted).T;ws=ws/self.width_scale;A=np.column_stack((np.ones(len(ys)),ys-c['width_ref_y']))
                coeff=np.linalg.lstsq(A,ws,rcond=None)[0]
                residual=abs(A@coeff-ws);keep=residual<max(5,float(np.median(residual))*2.5)
                if np.count_nonzero(keep)>=5:
                    coeff=np.linalg.lstsq(A[keep],ws[keep],rcond=None)[0]
                    # Bounded updates prevent corner fragments collapsing learned width.
                    ref=float(np.clip(coeff[0],self.wref*.9,self.wref*1.1))
                    slope=float(np.clip(coeff[1],self.wslope*.9,self.wslope*1.1))
                    alpha=c['width_update'];self.wref+=(ref-self.wref)*alpha;self.wslope+=(slope-self.wslope)*alpha
                    self.width_samples+=1
        self.observations=observations
        return observations

    def control(self,observations,dt):
        c=self.cfg;elapsed=max(.001,float(dt));dt=min(elapsed,.2);self.t+=elapsed
        good=sorted([p for p in observations if p is not None],key=lambda p:p['y'])
        fresh=len(good)>=c['min_valid_rows'] and good[-1]['y']-good[0]['y']>=8
        near_error=far_error=0.;near_x=far_x=c['target_x_px'];yn=yf=0.
        if fresh:
            self.last_seen=self.t
            ys=np.array([p['y'] for p in good]);xs=np.array([p['center'] for p in good])
            yn=float(np.clip(240*c['near_y_ratio'],ys[0],ys[-1]));yf=float(np.clip(240*c['far_y_ratio'],ys[0],ys[-1]))
            # Piecewise centerline keeps an actual fold; no one polynomial across a V.
            near_x=self.image_feature(good,yn);far_x=self.image_feature(good,yf)
            near_error=(near_x-c['target_x_px'])/max(self.width_at(yn)/2,12)
            far_error=(far_x-c['target_x_px'])/max(self.width_at(yf)/2,12)
            heading=far_error-near_error
            desired=c['gain_near']*near_error+c['gain_far']*far_error+c['gain_heading']*heading
            if self.feature_previous is not None:
                # Bound the measurement innovation itself, not just D. A single
                # implausible segmentation frame must not jump the P command.
                change=c['max_feature_rate']*elapsed
                desired=float(np.clip(desired,self.feature_previous-change,self.feature_previous+change))
            feature=desired
            raw_rate=0. if self.feature_previous is None else (feature-self.feature_previous)/elapsed
            raw_rate=float(np.clip(raw_rate,-c['max_feature_rate'],c['max_feature_rate']))
            self.feature_derivative+=(1-math.exp(-elapsed/c['derivative_tau']))*(raw_rate-self.feature_derivative)
            self.feature_previous=feature
            # Derivative of the visual steering feature adds phase lead in IMAGE
            # coordinates. It is not vehicle yaw/velocity or a world-space predictor.
            desired+=c['derivative_gain_sec']*self.feature_derivative
            desired=float(np.clip(desired,-c['max_steer'],c['max_steer']))
            filt=self.steer+(1-math.exp(-dt/c['control_tau']))*(desired-self.steer)
            self.steer=float(np.clip(filt,self.steer-c['steer_rate']*dt,self.steer+c['steer_rate']*dt))
            corner=abs(heading)>.18 or abs(far_error)>.3
            wanted=c['speed_corner'] if corner else c['speed_straight']
            self.speed=float(np.clip(wanted,self.speed-c['speed_ramp']*dt,self.speed+c['speed_ramp']*dt))
            double_rows=sum(p['source']=='double' for p in good)
            state=('SINGLE_EDGE' if double_rows==0 else
                   'MIXED_CORNER' if corner and double_rows<len(good) else
                   'MIXED' if double_rows<len(good) else 'CORNER' if corner else 'STRAIGHT')
        elif self.t-self.last_seen<=c['blind_bridge_sec']+1e-9 and self.result is not None:
            state='BRIDGE' # Keep last steering for a strictly bounded visual dropout.
        else:
            self.speed=self.steer=0.;state='LOST'
            self.feature_previous=None;self.feature_derivative=0.
            if self.t-self.last_seen>c['history_timeout']:self.previous=[];self.edge_history={}
        result=dict(speed=self.speed,steer=self.steer,state=state,error=near_x-c['target_x_px'],
            near_error=near_error,far_error=far_error,near_x=near_x,far_x=far_x,
            near_y=yn,far_y=yf,
            observations=observations,rows=self.rows,mask_y=self.mask_y,pixels=self.pixels,
            width_ref_px=self.wref,width_slope_px=self.wslope,width_samples=self.width_samples,
            width_scale=self.width_scale,
            valid_rows=len(good),double_rows=sum(p['source']=='double' for p in good),
            blind_sec=self.t-self.last_seen,connectors=self.connectors,
            compute_ms=0.)
        self.result=result;return result

    def step(self,mask,dt):
        started=time.perf_counter()
        mode='current_mask'
        if not self.armed:
            if np.count_nonzero(mask)>=18:self.warmup.append((mask>0).astype('uint8'))
            self.warmup=self.warmup[-self.cfg['warmup_frames']:]
            ready=len(self.warmup)>=self.cfg['warmup_frames']
            if ready:
                mode='stationary_warmup_vote'
                # The car is still commanded stationary: pixel voting is valid
                # here. Never pool unwarped images once the car is moving.
                votes=min(self.cfg['warmup_votes'],self.cfg['warmup_frames'])
                mask=(np.sum(self.warmup,axis=0)>=votes).astype('uint8')*255
                self.previous=[];self.edge_history={};self.feature_previous=None;self.feature_derivative=0.
            r=self.control(self.observe(mask),dt)
            self.armed=ready and r['valid_rows']>=self.cfg['min_valid_rows'] and r['near_y']>0
            if not self.armed:
                self.speed=self.steer=0.;r.update(speed=0.,steer=0.,state='ARMING')
        else:r=self.control(self.observe(mask),dt)
        r.update(armed=self.armed,warmup_count=len(self.warmup),input_mode=mode)
        r['compute_ms']=(time.perf_counter()-started)*1000;return r

    def overlay(self,image,mask,result):
        canvas=image.copy();color=np.zeros_like(canvas);color[mask>0]=(0,255,255)
        canvas=cv2.addWeighted(canvas,1.,color,.4,0)
        cv2.line(canvas,(int(self.cfg['target_x_px']),60),(int(self.cfg['target_x_px']),239),(130,130,130),1)
        pts=[]
        for p in result['observations']:
            if p is None:continue
            y=p['y'];cv2.line(canvas,(0,y),(319,y),(40,40,40),1)
            for name,col in [('left',(255,140,40)),('right',(255,140,40)),('center',(60,240,120))]:
                x=int(p[name]);
                if 0<=x<320:cv2.circle(canvas,(x,y),2,col,-1)
            pts.append([int(np.clip(p['center'],-500,800)),y])
        if len(pts)>1:cv2.polylines(canvas,[np.array(pts)],False,(70,230,110),1)
        for point,col in [('near',(50,200,255)),('far',(255,180,60))]:
            if result[point+'_y']:
                cv2.circle(canvas,(int(result[point+'_x']),int(result[point+'_y'])),5,col,1)
        cv2.putText(canvas,f'{result["state"]} {result["valid_rows"]}/{len(result["rows"])} PWM {1500+100*result["speed"]:.0f}',
                    (4,16),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,255,255),1)
        cv2.putText(canvas,f'near {result["near_error"]:+.2f} far {result["far_error"]:+.2f} s {result["steer"]:+.3f}',
                    (4,32),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,255,255),1)
        return canvas
