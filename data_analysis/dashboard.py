import altair as alt
import altair_transform
from IPython.display import HTML

alt.data_transformers.disable_max_rows()  # make altair plot DFs with more than 5000 rows

def make_dashboard(waf, pam, particle_snips, wafer_images, outPath):
    # ===================
    # Altair result plots
    # ===================

    # Interactive selections
    # ----------------------
    pickedX = alt.selection_single(encodings=['x'], init={'x': 'HCl'}, empty='none')  # , on='mouseover', clear='mouseout')  # selection of type "single" allows clicking of one element in one plot to filter other plots
    pickedY = alt.selection_single(encodings=['y'], init={'y': 'PA6'}, empty='none')  # , on='mouseover', clear='mouseout')  # selection of type "single" allows clicking of one element in one plot to filter other plots
    partpic = alt.selection_single(fields=['wafer', 'polymer', 'treatment', 'preIndex', 'postIndex'], empty='none', on='mouseover', nearest=True, init={'wafer': 'w20', 'polymer': 'PA6', 'treatment': 'HCl', 'preIndex': 85, 'postIndex': 13})
    modeSelector_radio = alt.binding_radio(options=['non_corrected', 'glm_corrected', 'water_corrected', 'glm_and_water_corrected'], name='Heatmap mode:   ')
    modeSelector = alt.selection_single(fields=['mode'], bind=modeSelector_radio, init={'mode': 'non_corrected'})
    treatSelector = alt.selection_multi(fields=['treatment'], bind='legend')
    polSelector = alt.selection_multi(fields=['polymer'], on='mouseover', clear='mouseout', nearest=True)
    particleprop_dropdown = alt.binding_select(options=['area', 'perimeter', 'intensity'], name='Particle property')
    particleprop_select = alt.selection_single(fields=['prop'], bind=particleprop_dropdown, init={'prop': 'area'})
    BDIslider = alt.binding_range(min=waf.BDI.min(), max=waf.BDI.max(), step=(waf.BDI.max() - waf.BDI.min()) / 100, name="show wafers with BDI ≤")
    BDIselector = alt.selection_single(fields=['BDIcutoff'], bind=BDIslider, init={'BDIcutoff': waf.BDI.max()})
    Nslider = alt.binding_range(min=waf.matched_count.min(), max=waf.matched_count.max(), step=1, name="show wafers with matched_count ≥")
    Nselector = alt.selection_single(fields=['Ncutoff'], bind=Nslider, init={'Ncutoff': waf.matched_count.min()})
    # prepostSelector_radio = alt.binding_radio(options=['pre_image', 'post_image'], name='Image:   ')
    # prepostSelector = alt.selection_single(fields=['key'], bind=prepostSelector_radio, init={'key': 'pre_image'})
    prepostSelector = alt.selection_single(fields=['key'], init={'key': 'pre_image'})  #, empty='init')



    # Quant Heatmap
    # -------------
    quantHM = alt.Chart(waf).mark_rect().encode(
        alt.X('treatment:N', sort=['water', 'H2O2', 'KOH', 'Pentane', 'SPT', 'HCl'], axis=alt.Axis(title=None, orient="top", domain=False)),
        opacity=alt.condition((alt.datum.pre_count >= Nselector.Ncutoff) & (alt.datum.BDI <= BDIselector.BDIcutoff), alt.value(1), alt.value(0))
    ).add_selection(
        modeSelector
    ).transform_filter(
        modeSelector
    ).properties(
        height=350, width=240
    ).add_selection(
        pickedY,
        pickedX
    )

    countHM = quantHM.encode(
        alt.Y('polymer:N', sort=['ABS', 'EVA', 'LDPE', 'PA6', 'PC', 'PET', 'PMMA', 'PP', 'PS', 'PVC', 'TPU', 'Acrylate', 'Epoxy'], axis=alt.Axis(title=None, orient="left", domain=False)),
        color=alt.Color(
            'particle_loss:Q',
            scale=alt.Scale(scheme='lightmulti', domain=(1, 0), clamp=True),  #color scheme similar to quali heatmap: 'lighttealblue', for +/- changes: 'redblue'
            sort='descending',
            legend=alt.Legend(format='%', title=['Loss', '[%]'], gradientLength=300, orient='left', titlePadding=20)),  # , gradientLabelOffset=10)),
        tooltip = alt.Tooltip(['wafer', 'pre_count', 'post_count', 'matched_count', 'particle_loss', 'BDI']),
    ).properties(
        title='Particle Numbers'
    )

    areaHM = quantHM.encode(
        alt.Y('polymer:N', sort=['ABS', 'EVA', 'LDPE', 'PA6', 'PC', 'PET', 'PMMA', 'PP', 'PS', 'PVC', 'TPU', 'Acrylate', 'Epoxy'], axis=alt.Axis(format='%', title=None, orient="left", domain=False, labels=False, ticks=False)),
        color=alt.Color(
            'area_change:Q',
            scale=alt.Scale(scheme='redblue', domain=(-1, 1), clamp=True),  #color scheme similar to quali heatmap: 'lighttealblue', for +/- changes: 'redblue'
            legend=alt.Legend(format='%', title=['Change', '+/- [%]'], gradientLength=300, orient='right', titlePadding=20)),  # , gradientLabelOffset=10)),
        tooltip = alt.Tooltip(['wafer', 'pre_area_matched', 'post_area_matched', 'area_change', 'BDI'])
    ).properties(
        title='Particle Areas'
    )

    # tileFrame = quantHM.encode(
    #     alt.X('treatment:N', sort=['water', 'H2O2', 'KOH', 'Pentane', 'SPT', 'HCl'], axis=alt.Axis(title=None, orient="top", domain=False), scale=alt.ScaleConfig(bandPaddingInner=0)),
    #     alt.Y('polymer:N', sort=['ABS', 'EVA', 'LDPE', 'PA6', 'PC', 'PET', 'PMMA', 'PP', 'PS', 'TPU', 'Acrylate', 'Epoxy'], axis=alt.Axis(title=None, orient="left", domain=False, labels=False, ticks=False), scale=alt.ScaleConfig(bandPaddingInner=0)),
    #     color=alt.condition(waferSel, alt.value('black'), alt.value('white'))
    # ).add_selection(
    #     waferSel
    # )


    # Wafer scatter
    # -------------
    waferScatter = alt.Chart(waf
    ).mark_point(filled=True).encode(
        alt.X('pre_count'),#, scale=alt.Scale(domain=(0,450))),
        alt.Y(alt.repeat('column'), type='quantitative', sort='ascending', axis=alt.Axis(format='%')),
        #alt.Y('particle_loss', sort='ascending'),
        color=alt.Color('treatment:N', legend=alt.Legend(orient='none', legendX=480, legendY=450)),
        size = alt.Size('BDI:Q', legend=alt.Legend(orient='none', legendX=-100, legendY=450), scale = alt.Scale(domain=(waf.BDI.min(), waf.BDI.max()))),
        tooltip=['wafer', 'polymer', 'treatment', 'pre_count', alt.Tooltip(alt.repeat('column'), type='quantitative'), 'BDI:Q'],
        #tooltip=['wafer', 'polymer', 'treatment', 'pre_count', 'particle_loss', 'BDI:Q'],
        opacity=alt.condition(treatSelector | polSelector, alt.value(1), alt.value(0.2))
    ).properties(
        height=200, width=200
    ).repeat(
        #row=['pre_count', 'matched_count', 'post_count'],
        column=['particle_loss', 'area_change']
    ).resolve_scale(
        y='independent'
    ).add_selection(
        treatSelector
    ).transform_filter(
        modeSelector
    ).add_selection(
        BDIselector
    ).transform_filter(
        alt.datum.BDI <= BDIselector.BDIcutoff
    ).add_selection(
        Nselector
    ).transform_filter(
        alt.datum.pre_count >= Nselector.Ncutoff
    ).add_selection(
        polSelector
    ).interactive(
    )

    # waferScatterRegLine = waferScatter.transform_calculate(
    #     particle_loss_NZ = 'datum.particle_loss + 0.01'
    # ).transform_regression('pre_count', 'particle_loss_NZ', method='linear'
    # ).mark_line(clip=True)

    # waferScatterRegParams =  waferScatter.transform_calculate(
    #     particle_loss_NZ = 'datum.particle_loss + 0.01'
    # ).transform_regression('pre_count', 'particle_loss_NZ', method='linear', params=True
    # ).mark_text(align='left', lineBreak='\n').encode(
    #     x=alt.value(100),  # pixels from left
    #     y=alt.value(20),  # pixels from top
    #     text='params:N'
    # ).transform_calculate(
    #     params='"r² = " + round(datum.rSquared * 100)/100 + "      y = " + round(datum.coef[0] * 10)/10 + " * x^" + round(datum.coef[1] * 10)/10'
    # )
    # #print(altair_transform.extract_data(waferScatterRegParams))

    # waferScatter = alt.layer(waferScatter, waferScatterRegLine, waferScatterRegParams
    # ).add_selection(
    #     treatSelector
    # ).transform_filter(
    #     modeSelector
    # ).add_selection(
    #     BDIselector
    # ).transform_filter(
    #     alt.datum.BDI <= BDIselector.BDIcutoff
    # ).add_selection(
    #     Nselector
    # ).transform_filter(
    #     alt.datum.pre_count >= Nselector.Ncutoff
    # ).add_selection(
    #     polSelector
    # ).interactive(
    # )


    # Wafer images
    # ------------
    Img = alt.Chart(wafer_images
    # ).transform_calculate(
    #     x='alt.value(1)',
    #     y='alt.value(1)'
    ).transform_fold(
        ['pre_image', 'post_image']
    ).mark_image(
        width=400,
        height=300
    ).encode(
        # x='x:Q',
        # y='y:Q',
        url = 'value:N'
    ).add_selection(
        prepostSelector
    ).transform_filter(
        prepostSelector
    ).transform_filter(
        pickedX & pickedY
    #).properties(
    #     width=400
    # ).interactive(
    )

    waferImageTitle = alt.Chart(wafer_images).transform_fold(
        ['pre_image', 'post_image']
    ).mark_text(dy=-220, angle=270, size=12).encode(
        text='label:N'
    ).transform_calculate(label='datum.treatment + " on " + datum.polymer + "  (" + datum.key + ")"'
    ).transform_filter(
        pickedX & pickedY
    ).transform_filter(
        prepostSelector
    )

    fullImg = Img + waferImageTitle


    # Particle Scatter plot
    # ------------
    particleScatter = alt.Chart(pam).mark_circle(size=100).encode(
        x=alt.X('preValue', axis=alt.Axis(title='Pre value of property')),
        y=alt.Y('postValue', axis=alt.Axis(title='Post value of property')),
        color=alt.condition(partpic, alt.value('orange'), alt.value('teal')),
        tooltip=alt.Tooltip(['preIndex', 'postIndex', 'preValue', 'postValue'])
    ).transform_filter(
        pickedX & pickedY
    # ).properties(title = alt.TitleParams('Particles of selcted wafer', orient='top')
    )

    waterScatter = alt.Chart(pam).mark_circle(color='grey', opacity=0.3).encode(
        x='preValue',
        y='postValue'
    ).transform_filter(
        {'and': [pickedY, alt.FieldEqualPredicate(field='treatment', equal='water')]}
    )

    particleLinReg = particleScatter.transform_regression('preValue', 'postValue', method="linear"
    ).mark_line(color="orange", clip=True)

    particleLinRegParams = particleScatter.transform_regression('preValue', 'postValue', method="linear", params=True
    ).mark_text(align='left', lineBreak='\n').encode(
        x=alt.value(20),  # pixels from left
        y=alt.value(20),  # pixels from top
        text='params:N'
    ).transform_calculate(
        params='"r² = " + round(datum.rSquared * 100)/100 + "      y = " + round(datum.coef[0] * 10)/10 + " + " + round(datum.coef[1] * 10)/10 + "x"'
    )

    waterLinReg = waterScatter.transform_regression('preValue', 'postValue',method="linear"
    ).mark_line(color="grey", clip=True)

    # band = alt.Chart(pam).mark_errorband(extent='ci').encode(
    #     x='preValue',
    #     y='postValue'
    # )

    identityLine = alt.Chart(pam).mark_line(color= 'black', strokeDash=[3,8], clip=True).encode(
        x=alt.X('preValue', axis=alt.Axis(title='')),
        y=alt.Y('preValue', axis=alt.Axis(title=''))
    )

    texts = alt.Chart().mark_text(dy=-180, size=12).encode(
        text='label:N'
    ).transform_calculate(label='datum.treatment + " on " + datum.polymer'
    ).transform_filter(
        pickedX & pickedY
    )


    # Boxplot
    # -------
    boxPlot = alt.Chart(pam).transform_calculate(
        Change='(datum.postValue / datum.preValue -1)'
    ).mark_boxplot(extent=0.5, outliers=True, clip=True).encode(
        x='treatment',
        y=alt.Y('Change:Q', axis=alt.Axis(format='%'),
                scale=alt.Scale(domain=(-1, 1)))
    ).transform_filter(
        particleprop_select
    ).transform_filter(
        {'and': [pickedY, {'or': [pickedX, alt.FieldEqualPredicate(field='treatment', equal='water')]}]}
    )


    # Particle snip images
    # --------------------
    snipPre = alt.Chart().mark_image(
        width=150,
        height=150
    ).encode(
        url = 'snip_pre:N'
    ).transform_filter(
        partpic
    ).properties(title = alt.TitleParams('Pre Treatment', orient='top')
    )


    snipPost = alt.Chart().mark_image(
        width=150,
        height=150
    ).encode(
        url = 'snip_post:N'
    ).transform_filter(
        partpic
    ).properties(title = alt.TitleParams('Post Treatment', orient='top')
    )


    # Putting plots together
    # ----------------------
    # wafer_side = alt.hconcat(tileFrame + countHM, tileFrame + areaHM).resolve_scale(
    #     color='independent'
    # )

    wafer_side = alt.vconcat(alt.hconcat(countHM, areaHM).resolve_scale(
        color='independent', y='independent', x='independent'), waferScatter)

    scatterAll = (particleScatter.add_selection(partpic) + particleLinReg + particleLinRegParams + waterScatter + waterLinReg + identityLine + texts
    ).add_selection(
        particleprop_select
    ).transform_filter(
        particleprop_select
    ).properties(
        width=300,
        height=300
    )

    snips = alt.vconcat(snipPre, snipPost, data=particle_snips)

    particle_side = alt.vconcat(alt.hconcat(scatterAll, boxPlot, snips, center=True), fullImg, center=True)

    alt.hconcat(wafer_side, particle_side,
                                    padding={"left": 50, "top": 50, "right": 50, "bottom": 50},
                                    spacing=50
    ).configure_scale(bandPaddingInner=0.1  # set space between heatmap tiles
    ).configure_title(orient='bottom', offset=20  # configure title of plot
    ).configure_view(
        strokeWidth=0  # get rid of chart box
    ).save(str(outPath / 'quant_results.html')
    )

    print(f"All done. Figure exported to {str(outPath / 'quant_results.html')}")
