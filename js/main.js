// Width and Height of the whole visualization
var margin = {top: 0, right: 20, bottom: 30, left: 50},
    width = 800 - margin.left - margin.right,
    height = 600 - margin.top - margin.bottom;


// Create SVG
var svg = d3.select("#map")
    .append( "svg" )
    .attr( "width", width + margin.left + margin.right )
    .attr( "height", height + margin.top + margin.bottom );

var g = svg.append( "g" );

// Set Projection Parameters
var albersProjection = d3.geo.albers()
    .scale( 190000 )
    .rotate( [71.057,0] )
    .center( [0, 42.313] )
    .translate( [width/2,height/2] );


var geoPath = d3.geo.path()
    .projection( albersProjection );

var tip = d3.tip()
    .attr('class', 'd3-tip')
    .offset([100, 0])
    .html(function (d) {

        var name = d.properties.Name;

        return "<span class ='toolTipHead'><b> " + name + "</b><br/>"
    })

svg.call(tip);


queue()
    .defer(d3.json, "data/neighborhoods_json.json")
    //.defer(d3.json, "data/d3_boston311_3K.json")
    .defer(d3.json, "data/clusters_calls.json")
    .await(function(error, data, clusters) {

//d3.json("data/neighborhoods_json.json", function(error, data) {
//    console.log(data.features)
//    console.log(clusters)
    g.selectAll( "path" )
        .data( data.features )
        .enter()
        .append( "path" )
        .attr( "fill", "#ccc" )
        .attr( "stroke", "#333")
        .attr( "d", geoPath )
        //.on("mouseout", function(d){
        //console.log(d.properties.Name)
        //})
        .on("mouseover", function (d) {
            tip.show(d)

        })
        .on("mouseout", function () {
            tip.hide()

        });

        //console.log(data.features)


        g.selectAll("circle")
            .data(clusters)
            .enter()
            .append("circle")
            .attr("cx", function(d) {
                return albersProjection([d.LONGITUDE, d.LATITUDE])[0];
            })
            .attr("cy", function(d) {
                return albersProjection([d.LONGITUDE, d.LATITUDE])[1];
            })
            .attr("r", 5)
            .style("fill", function(d){

                    return colors[ d.Cluster]
                })

            .style("opacity",0.7);

})

colors = ['#63ffc4 ','#0096b2 ','#ff7500 ']



//
//// Classic D3... Select non-existent elements, bind the data, append the elements, and apply attributes
//g.selectAll( "path" )
//    .data( "data/neighborhoods_json.features")
//    .enter()
//    .append( "path" )
//    .attr( "fill", "#ccc" )
//    .attr( "stroke", "#333")
//    .attr( "d", geoPath );
//
//console.log("data/neighborhoods_json.features")