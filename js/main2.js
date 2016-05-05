// Width and Height of the whole visualization
var margin = {top: 0, right: 20, bottom: 30, left: 50},
    width = 800 - margin.left - margin.right,
    height = 600 - margin.top - margin.bottom;

var selection="TYPE"
var features;
var ls_w
var ls_h
var point_clusters
// Create SVG
var svg2 = d3.select("#map2")
    .append( "svg" )
    .attr( "width", width + margin.left + margin.right )
    .attr( "height", height + margin.top + margin.bottom );

var g2 = svg2.append( "g" );

// Set Projection Parameters
var albersProjection = d3.geo.albers()
    .scale( 190000 )
    .rotate( [71.057,0] )
    .center( [0, 42.313] )
    .translate( [width/1.5,height/2] );


var geoPath = d3.geo.path()
    .projection( albersProjection );

var tip = d3.tip()
    .attr('class', 'd3-tip')
    .offset([100, 0])
    .html(function (d) {

        var name = d.properties.Name;

        return "<span class ='toolTipHead'><b> " + name + "</b><br/>"
    })

svg2.call(tip);


queue()
    .defer(d3.json, "data/neighborhoods_json.json")
    .defer(d3.json, "data/unclustered.json")
    .defer(d3.json, "data/d3_boston311_3K.json")


    .await(function(error, data, data_info, clusters) {


        g2.selectAll("path")
            .data(data.features)
            .enter()
            .append("path")
            .attr("fill", "#ccc")
            .attr("stroke", "#333")
            .attr("d", geoPath)
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


        var unique = []
        var count = 0;

        data_info.forEach(function (d) {
            if (unique.indexOf(d[selection]) === -1) {
                unique.push(d[selection])
                count = count + 1;
            }
        });

        point_clusters = []
        clusters.forEach(function (d) {

            point_clusters.push(d.cluster)
        });


        g2.selectAll("circle")
            .data(data_info)
            .enter()
            .append("circle")
            .attr("cx", function (d) {
                return albersProjection([d.LONGITUDE, d.LATITUDE])[0];
            })
            .attr("cy", function (d) {
                return albersProjection([d.LONGITUDE, d.LATITUDE])[1];
            })
            .attr("r", 5)
            .style("fill", function (d) {
                idx = unique.indexOf(d[selection])
                return colorbrewer.Set4[12][idx]
            })

            .style("opacity", 0.7)
            //.style("stroke", function(d,i){
            //    return colorbrewer.Set1[4][ point_clusters[i]]
            //})
            //.style("stroke-width", 2)


        var legend = svg2.selectAll("g.legend")
            .data(unique)
            .enter()
            .append("g")
            .attr("class", "legend");

        ls_w = 20
        ls_h = 20;

        legend.append("rect")

        legend.append("text")

        features = data_info;

        updateMap()
    });


d3.select("#ranking-type").on("change", function() {
    selection = d3.select("#ranking-type").property("value")
    console.log(selection)
    updateMap()
})

function updateMap() {
    var unique = []

    features.forEach(function (d) {
        if (unique.indexOf(d[selection]) === -1) {
            unique.push(d[selection])
        }
    });


    var circle = g2.selectAll("circle")
        .data(features)

    circle.enter()
        .append("circle")

    circle
        .transition()
        .transition(800)
        .style("fill", function (d) {
            idx = unique.indexOf(d[selection])
            return colorbrewer.Set4[12][idx]
        })

    circle.exit()
        .transition()
        .duration(800)
        .remove();

    var legend = svg2.select("g.legend")

    var legendBox = legend.selectAll("rect")
        .data(unique);

    legendBox.enter()
        .append("rect");

    legendBox.exit()
        .remove();

    legendBox
        .attr("x", 10)
        .attr("y", function(d, i){
            return (i*ls_h) + 2*ls_h;})
        .attr("width", ls_w)
        .attr("height", ls_h)
        .style("fill", function(d, i) {
            return colorbrewer.Set4[12][i]; })
        .style("opacity", 0.8);

    var legendText = legend.selectAll("text")
        .data(unique);

    legendText.enter()
        .append("text");

    legendText.exit()
        .remove();

    legendText
        .attr("x", 50)
        .attr("y", function(d, i){ return (i*ls_h) + 2*ls_h +12;})
        .text(function(d, i){ return unique[i]; });


}

//
//    var legend = svg2.selectAll("g.legend")
//        .data(unique)
//        .enter().append("g")
//        .attr("class", "legend");
//
//    var ls_w = 20, ls_h = 20;
//
//    legend.append("rect")
//        .attr("x", 20)
//        .attr("y", function(d, i){ return height - (i*ls_h) - 2*ls_h;})
//        .attr("width", ls_w)
//        .attr("height", ls_h)
//        .style("fill", function(d, i) { return colorbrewer.Set4[12][i]; })
//        .style("opacity", 0.8);
//
//    legend.append("text")
//        .attr("x", 50)
//        .attr("y", function(d, i){ return height - (i*ls_h) - ls_h - 4;})
//        .text(function(d, i){ return unique[i]; });
//}

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