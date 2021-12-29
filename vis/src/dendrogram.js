
import React from "react"
import * as d3 from 'd3'
import axios from 'axios'

export default class Dendrogram extends React.PureComponent {
  constructor(props) {
    super(props)
    this.width = 500
    this.height = 500
    this.margin = 20
    this.trace = []
    this.state = {
      num_clu: 2,
    }
  }
  
  componentDidMount(){
    this.getData(
      "http://127.0.0.1:5000/get_root",
      null,
      (res)=>{
        const ret = res.data
        const decor = (v, i) => [v, i];          // set index to value
        const undecor = a => a[1];               // leave only index
        const argsort = arr => arr.map(decor).sort((a,b)=>a[0]-b[0]).map(undecor)
        const order = argsort(ret.choice)
        this.denData = ret.clu 
        this.choice = order.map(v=>ret.choice[v])
        this.tsne_data = order.map(v=>ret.tsne[v])
        this.tsne_num = this.choice.length
        this.selected = Array(this.tsne_num).fill(true)
        this.draw()
      })
  }
  componentDidUpdate(){
    this.draw()
  }

  getData(url,para,fn){
      axios.get(url,{
              params: para,
          })
          .then(fn);
  }
  
  draw = ()=>{
    let choice = this.choice
    let tsne_data = this.tsne_data
    let denData = this.denData
    // update selection
    function remove_selection(node){
      node.selected = false
      for (const child of node.children)
        remove_selection(child)
    }
    function trace_node(denData, trace){
      let cur = denData
      for (const t of trace){
        cur = cur.children[t]
      }
      return cur
    }
    remove_selection(denData)
    trace_node(denData,this.trace).selected = true

    // Create the cluster layout:
    var cluster = d3.cluster()
      .size([this.width - 2 * this.margin, this.height - 2 * this.margin]); 

    // Give the data to this cluster layout:
    var root = d3.hierarchy(denData, function (d) {
        return d.children;
    });
    cluster(root)

    // Features of the links between nodes:
    var linksGenerator = d3.linkVertical()
      .x(d => d.x)
      .y(d => d.y)

    // Add the links between nodes:
    let g = d3.select(this.g)

    g.selectAll('path')
      .data(root.links())
      .join('path')
      .attr("d", linksGenerator)
      .style("fill", 'none')
      .attr("stroke", 'black')
      // .style('visibility', (d) => {
      //   return d.target.height >= 4 ? 'visible' : 'hidden';
      // })

    function get_trace(d) {
      let trace = []
      let cur = d
      while(cur.parent){
        for (let i=0;i< cur.parent.children.length;i++){
          if (cur===cur.parent.children[i]) {
            trace.unshift(i)
            break
          }
        }
        cur = cur.parent
      }
      return trace
    }
    function ordered_list_intersection(list1,list2){
      let mask = Array(list1.length).fill(false)
      list1.sort((a,b)=>a-b)
      list2.sort((a,b)=>a-b)
      let cur1= 0, cur2=0
      while (true){
        if (list1[cur1]<list2[cur2]){
          if (cur1 === list1.length - 1) break
          else cur1 ++
        }
        else if (list1[cur1]===list2[cur2]){
          mask[cur1] = true
          if (cur1 === list1.length - 1 || cur2 === list2.length - 1)
            break
          else{
            cur1 ++
            cur2 ++
          }
        }
        else {
          if (cur2 === list2.length - 1) break
          else cur2 ++
        }
      }
      return mask
    }

    g.selectAll("circle")
      .data(root.descendants())
      .join("circle")
      .attr('cx',(d)=>d.x)
      .attr('cy',(d)=>d.y)
      .attr('r',10)
      .style('stroke','black')
      .style('stroke-width','1px')
      .style("fill", (d,i)=>d.data.selected?"#fc8d62":"#69b3a2")
      .on('contextmenu',(e,d,i)=>{
        e.preventDefault()
        this.trace = get_trace(d)
        if (d.data.children.length > 0){
          let cur = trace_node(denData,this.trace)
          cur.children = []
          this.draw()
          this.getData(
            "http://127.0.0.1:5000/rm_children",
            {
              trace: this.trace,
            },null)
        }
        else {
          this.getData(
            "http://127.0.0.1:5000/get_children",
            {
              trace: this.trace,
              num_clu: this.state.num_clu,
            },
            (res)=>{
              let node = trace_node(denData,this.trace)
              node.children = res.data
              this.draw()
            })
        }
      })
      .on('click',(e,d,i)=>{
        let list1 = [...choice]
        let list2 = [...d.data.idx]
        this.selected = ordered_list_intersection(list1,list2)
        this.trace = get_trace(d)
        this.draw()
        this.getData(
          "http://127.0.0.1:5000/vtk",{'trace':this.trace},null)
      })
  

    // tsne 
    this.max_x= -1e10
    this.max_y= -1e10
    this.min_x= 1e20
    this.min_y= 1e20
    for (let d of tsne_data){
      this.max_x = this.max_x > d[0] ? this.max_x : d[0]
      this.max_y = this.max_y > d[1] ? this.max_y : d[1]
      this.min_x = this.min_x < d[0] ? this.min_x : d[0]
      this.min_y = this.min_y < d[1] ? this.min_y : d[1]
    }
    let x = d3.scaleLinear()
      .domain([this.min_x,this.max_x])
      .range([0,this.width - 2 * this.margin])
    let y = d3.scaleLinear()
      .domain([this.min_y,this.max_y])
      .range([0,this.height - 2 * this.margin])
    d3.select(this.gtsne)
      .selectAll("circle")
      .data(tsne_data)
      .join("circle")
      .attr("cx", (d) => {
          return x(d[0]) 
        } )
      .attr("cy", function (d) { return y(d[1]); } )
      .attr("r", 2)
      .style("fill", (_,i) => {
          if (this.selected[i]) return "#d95f02"
          else return "steelblue"
      })
  }
  handleChange = (e)=>{
    this.setState({num_clu:e.target.value})
  }

  render(){
    return(
      <div>
          <svg
              ref={r => this.svg = r}
              viewBox={`0 0 ${this.width} ${this.height}`}
              width = "45%" >
            <rect x={5} y={5} width={this.width-10} height={this.height-10} rx="15" style={{fill:'#f0f0f0', strokeWidth:4, stroke:'black'}} />
            <g ref={r=>this.g=r} transform={`translate(${this.margin},${this.margin})`}/>
            <foreignObject x="10" y="10" width="200" height="100">
              <label htmlFor="num_clu"># clusters:</label>
              <input 
                type="text" id="num_clu" name="num_clu" 
                style={{width:'50%',marginLeft:'5px'}} value={this.state.num_clu}
                onChange={this.handleChange}
              />
            </foreignObject>
          </svg>
          <svg
              ref={r => this.tsne = r}
              viewBox={`0 0 ${this.width} ${this.height}`}
              width = "45%" >
            <rect x={5} y={5} width={this.width-10} height={this.height-10} rx="15" style={{fill:'#f0f0f0', strokeWidth:4, stroke:'black'}} />
            <g ref={r=>this.gtsne=r} transform={`translate(${this.margin},${this.margin})`}/>
          </svg>
      </div>
    )
  }
}

